# ==============================================================
#  Robust and Reproducible Prediction of HDL Cholesterol Under Outcome Perturbation:
#  (An NHANES Machine Learning Study)
#
#  UPDATED PIPELINE (Reviewer-driven improvements):
#  ✅ Systematic pre-model feature screening:
#     (1) near-zero variance removal (post-bake)
#     (2) multicollinearity reduction via correlation filtering
#     (3) FULL vs SCREENED feature-set comparison (same CV protocol)
#
#  Notes:
#  - Public TEST set is UNLABELED (no LBDHDD_outcome)
#  - Report performance via CV (OOF) on TRAIN only
#  - Sensitivity to additional outcome noise (sigma grid)
#  - Compare: Linear + Penalized + Tree/Boosting
#  - Feature stability under perturbation (Elastic-net; bootstrap)
#  - Generate BLIND predictions on TEST and save for submission
#
#  ROOT OUTPUT DIRECTORY:
#    C:/Users/kokye/OneDrive/Desktop/Research1/Project
#
#  Outputs:
#    tables/*.csv
#    figures/*.png
#    rds/*.rds
#    sessionInfo.txt
# ==============================================================

rm(list = ls())
set.seed(2026)

# ---------------------------
# 0) Packages
# ---------------------------
needed <- c(
  "tidyverse", "yardstick", "glmnet", "recipes",
  "broom", "readr", "fs", "glue", "patchwork",
  "scales", "RColorBrewer",
  "ranger", "xgboost",
  "caret"   # <-- for nearZeroVar + findCorrelation (screening)
)

to_install <- setdiff(needed, rownames(installed.packages()))
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

library(tidyverse)
library(yardstick)
library(glmnet)
library(recipes)
library(broom)
library(readr)
library(fs)
library(glue)
library(patchwork)
library(scales)
library(RColorBrewer)
library(ranger)
library(xgboost)
library(caret)

# Force tidyverse verbs (prevents masking issues)
select    <- dplyr::select
filter    <- dplyr::filter
mutate    <- dplyr::mutate
arrange   <- dplyr::arrange
summarise <- dplyr::summarise
rename    <- dplyr::rename

# ---------------------------
# 0.1) Global Theme + Colors
# ---------------------------
theme_set(theme_minimal(base_size = 13))

COL <- list(
  blue   = "#2C7FB8",
  red    = "#F03B20",
  green  = "#31A354",
  purple = "#756BB1",
  gray   = "grey35",
  black  = "grey10"
)

# ---------------------------
# 1) Root Directory
# ---------------------------
root_dir <- "C:/Users/kokye/OneDrive/Desktop/Research1/RPROJECT"

dir_create(root_dir)
dir_create(path(root_dir, "tables"))
dir_create(path(root_dir, "figures"))
dir_create(path(root_dir, "rds"))

save_tbl <- function(df, name) write_csv(df, path(root_dir, "tables", paste0(name, ".csv")))
save_rds <- function(obj, name) saveRDS(obj, path(root_dir, "rds", paste0(name, ".rds")))

save_plot <- function(p, name, w = 8, h = 5, dpi = 320) {
  print(p)
  ggsave(
    filename = path(root_dir, "figures", paste0(name, ".png")),
    plot = p, width = w, height = h, dpi = dpi
  )
}

save_combined_plot <- function(p, name, w = 18, h = 14, dpi = 320) {
  print(p)
  ggsave(
    filename = path(root_dir, "figures", paste0(name, ".png")),
    plot = p, width = w, height = h, dpi = dpi
  )
}

show_tbl <- function(df, title = NULL, n = 10, view = interactive()) {
  if (!is.null(title)) cat("\n----- ", title, " -----\n", sep = "")
  print(as_tibble(df) %>% head(n))
  if (view) try(View(df), silent = TRUE)
}

# ==============================================================
# 2) Data (NHANES-derived ASASF)
# ==============================================================

tmp <- tempfile()

download.file("https://luminwin.github.io/ASASF/train.rds", tmp, mode = "wb")
train <- readRDS(tmp)

labels_list <- lapply(train, attr, "label")
save_rds(labels_list, "variable_labels")

download.file("https://luminwin.github.io/ASASF/test.rds", tmp, mode = "wb")
test <- readRDS(tmp)

outcome <- "LBDHDD_outcome"
stopifnot(outcome %in% names(train))   # outcome only in TRAIN
stopifnot(nrow(train) == 1000, nrow(test) == 200)

# Remove label attributes (keeps values)
strip_labels <- function(df) {
  df %>% mutate(across(everything(), ~ { attr(.x, "label") <- NULL; .x }))
}
train <- strip_labels(train)
test  <- strip_labels(test)

# --- Table: Outcome descriptive summary (TRAIN only) ---
tbl_outcome <- train %>%
  summarise(
    n = n(),
    mean = mean(.data[[outcome]], na.rm = TRUE),
    sd   = sd(.data[[outcome]], na.rm = TRUE),
    min  = min(.data[[outcome]], na.rm = TRUE),
    q1   = quantile(.data[[outcome]], 0.25, na.rm = TRUE),
    med  = median(.data[[outcome]], na.rm = TRUE),
    q3   = quantile(.data[[outcome]], 0.75, na.rm = TRUE),
    max  = max(.data[[outcome]], na.rm = TRUE)
  )

save_tbl(tbl_outcome, "Table1_outcome_summary_train")
show_tbl(tbl_outcome, "Table 1. Outcome Summary (Train)", n = 20)

# --- Missingness overview (TRAIN) ---
miss_tbl <- train %>%
  summarise(across(everything(), ~ mean(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_rate") %>%
  arrange(desc(missing_rate))
save_tbl(miss_tbl, "missingness_train")
show_tbl(miss_tbl, "Missingness (Top 20 variables)", n = 20)

# ==============================================================
# 3) Preprocessing (recipes) — fit on TRAIN, apply to TRAIN+TEST
# ==============================================================

rec <- recipe(as.formula(paste(outcome, "~ .")), data = train) %>%
  step_zv(all_predictors()) %>%                      # removes zero-variance BEFORE dummies
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors())

rec_prep <- prep(rec, training = train, verbose = FALSE)

train_baked <- bake(rec_prep, new_data = train)
test_baked  <- bake(rec_prep, new_data = test)

x_train_full <- train_baked %>%
  dplyr::select(-dplyr::all_of(outcome)) %>%
  as.matrix()

y0 <- train_baked[[outcome]]
x_test_full <- test_baked %>% as.matrix()

save_rds(list(
  p_full = ncol(x_train_full),
  n_train = nrow(x_train_full),
  n_test = nrow(x_test_full),
  baked_names_full = colnames(x_train_full)
), "design_matrix_info_full")

# ==============================================================
# 3.1) NEW: Systematic Feature Screening (Reviewer suggestion)
#  (1) Near-zero variance features (post-bake, after dummies)
#  (2) Multicollinearity reduction via correlation filter
# ==============================================================
apply_feature_screening <- function(x_train, x_test, nzv_freq_cut = 95/5, nzv_unique_cut = 10,
                                    cor_cutoff = 0.90) {
  stopifnot(ncol(x_train) == ncol(x_test))
  
  # --- (1) Near-zero variance (post-bake)
  nzv_idx <- nearZeroVar(x_train,
                         freqCut = nzv_freq_cut,
                         uniqueCut = nzv_unique_cut)
  
  keep1 <- setdiff(seq_len(ncol(x_train)), nzv_idx)
  x1_tr <- x_train[, keep1, drop = FALSE]
  x1_te <- x_test[,  keep1, drop = FALSE]
  
  removed_nzv <- colnames(x_train)[nzv_idx]
  
  # --- (2) Correlation filter (multicollinearity)
  # Use complete obs (no NAs after imputation); if p is big, this can still run.
  cor_mat <- suppressWarnings(cor(x1_tr))
  cor_mat[is.na(cor_mat)] <- 0
  
  high_cor_idx <- findCorrelation(cor_mat, cutoff = cor_cutoff, names = FALSE, exact = TRUE)
  keep2 <- setdiff(seq_len(ncol(x1_tr)), high_cor_idx)
  
  x2_tr <- x1_tr[, keep2, drop = FALSE]
  x2_te <- x1_te[, keep2, drop = FALSE]
  
  removed_cor <- colnames(x1_tr)[high_cor_idx]
  
  list(
    x_train_screened = x2_tr,
    x_test_screened  = x2_te,
    removed_nzv      = removed_nzv,
    removed_cor      = removed_cor,
    p_screened       = ncol(x2_tr)
  )
}

screen <- apply_feature_screening(x_train_full, x_test_full, cor_cutoff = 0.90)

x_train_scr <- screen$x_train_screened
x_test_scr  <- screen$x_test_screened

save_rds(list(
  p_full = ncol(x_train_full),
  p_screened = screen$p_screened,
  removed_nzv = screen$removed_nzv,
  removed_cor = screen$removed_cor
), "feature_screening_summary")

save_tbl(tibble(feature = screen$removed_nzv), "screen_removed_near_zero_variance")
save_tbl(tibble(feature = screen$removed_cor), "screen_removed_high_correlation")

# ==============================================================
# 4) Noise Sensitivity Design (additional synthetic noise)
#    y^(sigma) = y0 + eps, eps ~ N(0, sigma^2)
# ==============================================================

sigma_grid <- c(0, 0.5, 1, 2, 3, 5)

make_noisy_y <- function(y, sigma, seed = 2026) {
  set.seed(seed + round(1000 * sigma))
  y + rnorm(length(y), mean = 0, sd = sigma)
}

y_list <- map(sigma_grid, ~ make_noisy_y(y0, .x))
names(y_list) <- paste0("sigma_", sigma_grid)
save_rds(list(sigma_grid = sigma_grid, y_list = y_list), "noise_scenarios")

# ==============================================================
# 5) CV Folds (fixed across experiments for fair comparison)
# ==============================================================

K <- 10
set.seed(2026)
fold_id <- sample(rep(1:K, length.out = length(y0)))

# metric helper
calc_metrics <- function(truth, pred) {
  tibble(
    RMSE = rmse_vec(truth, pred),
    MAE  = mae_vec(truth, pred),
    R2   = rsq_vec(truth, pred)
  )
}

# ==============================================================
# 6) Tuning Helpers (baseline sigma=0 only; per feature set)
# ==============================================================

tune_glmnet_baseline <- function(x_train, y0, K, seed = 2026) {
  set.seed(seed)
  cv_ridge <- cv.glmnet(x_train, y0, alpha = 0, nfolds = K, standardize = FALSE)
  ridge_lambda_1se <- cv_ridge$lambda.1se
  
  set.seed(seed)
  cv_lasso <- cv.glmnet(x_train, y0, alpha = 1, nfolds = K, standardize = FALSE)
  lasso_lambda_1se <- cv_lasso$lambda.1se
  
  alphas <- seq(0.1, 0.9, by = 0.1)
  set.seed(seed)
  cv_en_tbl <- map_dfr(alphas, function(a) {
    fit <- cv.glmnet(x_train, y0, alpha = a, nfolds = K, standardize = FALSE)
    tibble(alpha = a, cvm_min = min(fit$cvm), lambda_1se = fit$lambda.1se, fit = list(fit))
  })
  best_row <- cv_en_tbl %>% arrange(cvm_min) %>% slice(1)
  best_alpha <- best_row$alpha
  cv_en_best <- best_row$fit[[1]]
  enet_lambda_1se <- best_row$lambda_1se
  
  list(
    cv_ridge = cv_ridge,
    cv_lasso = cv_lasso,
    cv_en_tbl = cv_en_tbl,
    best_alpha = best_alpha,
    cv_en_best = cv_en_best,
    ridge_lambda_1se = ridge_lambda_1se,
    lasso_lambda_1se = lasso_lambda_1se,
    enet_lambda_1se  = enet_lambda_1se
  )
}

fit_rf <- function(x_tr, y_tr, mtry, seed = 2026) {
  dat <- as.data.frame(x_tr)
  dat$y <- y_tr
  ranger(
    y ~ ., data = dat,
    num.trees = 500,
    mtry = mtry,
    importance = "permutation",
    seed = seed
  )
}

tune_rf_baseline <- function(x_train, y0, fold_id, K, seed = 2026) {
  rf_tune_grid <- tibble(
    mtry = unique(pmax(2, round(c(sqrt(ncol(x_train)), ncol(x_train)/10, ncol(x_train)/5))))
  ) %>% distinct() %>% arrange(mtry)
  
  oof_rf_for_mtry <- function(mtry) {
    pred <- rep(NA_real_, length(y0))
    for (k in 1:K) {
      idx_te <- which(fold_id == k)
      idx_tr <- setdiff(seq_along(y0), idx_te)
      rf_fit <- fit_rf(x_train[idx_tr, , drop = FALSE], y0[idx_tr], mtry = mtry, seed = seed + k)
      pred[idx_te] <- predict(rf_fit, data = as.data.frame(x_train[idx_te, , drop = FALSE]))$predictions
    }
    calc_metrics(y0, pred)
  }
  
  rf_tune_tbl <- rf_tune_grid %>%
    mutate(metrics = map(mtry, oof_rf_for_mtry)) %>%
    unnest(metrics) %>%
    arrange(RMSE)
  
  best_mtry <- rf_tune_tbl %>% slice(1) %>% pull(mtry)
  
  list(best_mtry = best_mtry, rf_tune_tbl = rf_tune_tbl)
}

xgb_fit <- function(x_tr, y_tr, params, nrounds) {
  dtrain <- xgb.DMatrix(x_tr, label = y_tr)
  xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
}

tune_xgb_baseline <- function(x_train, y0, fold_id, K, seed = 2026) {
  xgb_grid <- tidyr::expand_grid(
    max_depth = c(3, 6),
    eta = c(0.03, 0.08),
    subsample = c(0.8),
    colsample_bytree = c(0.8),
    nrounds = c(300, 600)
  )
  
  oof_xgb_for_row <- function(row) {
    pred <- rep(NA_real_, length(y0))
    params <- list(
      objective = "reg:squarederror",
      max_depth = row$max_depth,
      eta = row$eta,
      subsample = row$subsample,
      colsample_bytree = row$colsample_bytree
    )
    for (k in 1:K) {
      idx_te <- which(fold_id == k)
      idx_tr <- setdiff(seq_along(y0), idx_te)
      fit <- xgb_fit(x_train[idx_tr, , drop = FALSE], y0[idx_tr], params, nrounds = row$nrounds)
      pred[idx_te] <- predict(fit, newdata = x_train[idx_te, , drop = FALSE])
    }
    calc_metrics(y0, pred)
  }
  
  xgb_tune_tbl <- xgb_grid %>%
    mutate(
      metrics = purrr::pmap(
        list(max_depth, eta, subsample, colsample_bytree, nrounds),
        function(max_depth, eta, subsample, colsample_bytree, nrounds) {
          row <- tibble(
            max_depth = max_depth,
            eta = eta,
            subsample = subsample,
            colsample_bytree = colsample_bytree,
            nrounds = nrounds
          )
          oof_xgb_for_row(row)
        }
      )
    ) %>%
    unnest(metrics) %>%
    arrange(RMSE)
  
  best_xgb <- xgb_tune_tbl %>% slice(1)
  list(best_xgb = best_xgb, xgb_tune_tbl = xgb_tune_tbl)
}

# ==============================================================
# 6A) Tune BASELINE for FULL feature set
# ==============================================================
tune_full_glmnet <- tune_glmnet_baseline(x_train_full, y0, K)
tune_full_rf     <- tune_rf_baseline(x_train_full, y0, fold_id, K)
tune_full_xgb    <- tune_xgb_baseline(x_train_full, y0, fold_id, K)

save_rds(tune_full_glmnet, "tuning_glmnet_baseline_FULL")
save_rds(tune_full_rf,     "tuning_rf_baseline_FULL")
save_rds(tune_full_xgb,    "tuning_xgb_baseline_FULL")

save_tbl(tune_full_rf$rf_tune_tbl,   "rf_tuning_baseline_FULL")
save_tbl(tune_full_xgb$xgb_tune_tbl, "xgb_tuning_baseline_FULL")

# ==============================================================
# 6B) Tune BASELINE for SCREENED feature set (Reviewer suggestion)
# ==============================================================
tune_scr_glmnet <- tune_glmnet_baseline(x_train_scr, y0, K)
tune_scr_rf     <- tune_rf_baseline(x_train_scr, y0, fold_id, K)
tune_scr_xgb    <- tune_xgb_baseline(x_train_scr, y0, fold_id, K)

save_rds(tune_scr_glmnet, "tuning_glmnet_baseline_SCREENED")
save_rds(tune_scr_rf,     "tuning_rf_baseline_SCREENED")
save_rds(tune_scr_xgb,    "tuning_xgb_baseline_SCREENED")

save_tbl(tune_scr_rf$rf_tune_tbl,   "rf_tuning_baseline_SCREENED")
save_tbl(tune_scr_xgb$xgb_tune_tbl, "xgb_tuning_baseline_SCREENED")

# ==============================================================
# 7) OOF CV under Noise (fixed tuning per feature set)
# ==============================================================

fit_ols_fold <- function(x_tr, y_tr) {
  dat <- as.data.frame(x_tr)
  dat$y <- y_tr
  lm(y ~ ., data = dat)
}
pred_ols <- function(fit, x_te) predict(fit, newdata = as.data.frame(x_te))

fit_glmnet_fold <- function(x_tr, y_tr, alpha) {
  glmnet(x_tr, y_tr, alpha = alpha, standardize = FALSE)
}
pred_glmnet <- function(fit, x_te, lambda) as.numeric(predict(fit, newx = x_te, s = lambda))

pred_rf <- function(fit, x_te) predict(fit, data = as.data.frame(x_te))$predictions

fit_xgb_fold <- function(x_tr, y_tr, best_xgb_row) {
  params <- list(
    objective = "reg:squarederror",
    max_depth = best_xgb_row$max_depth,
    eta = best_xgb_row$eta,
    subsample = best_xgb_row$subsample,
    colsample_bytree = best_xgb_row$colsample_bytree
  )
  xgb_fit(x_tr, y_tr, params = params, nrounds = best_xgb_row$nrounds)
}
pred_xgb <- function(fit, x_te) predict(fit, newdata = x_te)

run_oof_all_models <- function(x_train, y, sigma_label, tune) {
  n <- length(y)
  
  pred_ols_v    <- rep(NA_real_, n)
  pred_ridge_v  <- rep(NA_real_, n)
  pred_lasso_v  <- rep(NA_real_, n)
  pred_enet_v   <- rep(NA_real_, n)
  pred_rf_v     <- rep(NA_real_, n)
  pred_xgb_v    <- rep(NA_real_, n)
  
  for (k in 1:K) {
    idx_te <- which(fold_id == k)
    idx_tr <- setdiff(seq_along(y), idx_te)
    
    xtr <- x_train[idx_tr, , drop = FALSE]
    ytr <- y[idx_tr]
    xte <- x_train[idx_te, , drop = FALSE]
    
    # OLS
    fit_ols_k <- fit_ols_fold(xtr, ytr)
    pred_ols_v[idx_te] <- pred_ols(fit_ols_k, xte)
    
    # Ridge/Lasso/ENet (fixed lambda.1se from baseline per feature set)
    fit_ridge_k <- fit_glmnet_fold(xtr, ytr, alpha = 0)
    pred_ridge_v[idx_te] <- pred_glmnet(fit_ridge_k, xte, tune$glmnet$ridge_lambda_1se)
    
    fit_lasso_k <- fit_glmnet_fold(xtr, ytr, alpha = 1)
    pred_lasso_v[idx_te] <- pred_glmnet(fit_lasso_k, xte, tune$glmnet$lasso_lambda_1se)
    
    fit_enet_k <- fit_glmnet_fold(xtr, ytr, alpha = tune$glmnet$best_alpha)
    pred_enet_v[idx_te] <- pred_glmnet(fit_enet_k, xte, tune$glmnet$enet_lambda_1se)
    
    # Random Forest (fixed mtry per feature set)
    rf_fit_k <- fit_rf(xtr, ytr, mtry = tune$rf$best_mtry, seed = 2026 + k)
    pred_rf_v[idx_te] <- pred_rf(rf_fit_k, xte)
    
    # XGBoost (fixed best row per feature set)
    xgb_fit_k <- fit_xgb_fold(xtr, ytr, best_xgb_row = tune$xgb$best_xgb)
    pred_xgb_v[idx_te] <- pred_xgb(xgb_fit_k, xte)
  }
  
  enet_label <- glue("ENet(a={round(tune$glmnet$best_alpha, 2)})")
  rf_label   <- glue("RF (mtry={tune$rf$best_mtry})")
  
  dplyr::bind_rows(
    tibble(sigma = sigma_label, model = "OLS",        calc_metrics(y, pred_ols_v)),
    tibble(sigma = sigma_label, model = "Ridge",      calc_metrics(y, pred_ridge_v)),
    tibble(sigma = sigma_label, model = "Lasso",      calc_metrics(y, pred_lasso_v)),
    tibble(sigma = sigma_label, model = enet_label,   calc_metrics(y, pred_enet_v)),
    tibble(sigma = sigma_label, model = rf_label,     calc_metrics(y, pred_rf_v)),
    tibble(sigma = sigma_label, model = "XGB (best)", calc_metrics(y, pred_xgb_v))
  )
}

tune_FULL <- list(glmnet = tune_full_glmnet, rf = tune_full_rf, xgb = tune_full_xgb)
tune_SCR  <- list(glmnet = tune_scr_glmnet,  rf = tune_scr_rf,  xgb = tune_scr_xgb)

# --- FULL features performance table
noise_perf_full <- purrr::imap_dfr(y_list, ~ run_oof_all_models(x_train_full, .x, .y, tune_FULL)) %>%
  mutate(sigma_num = readr::parse_number(sigma), feature_set = "FULL")

# --- SCREENED features performance table
noise_perf_scr <- purrr::imap_dfr(y_list, ~ run_oof_all_models(x_train_scr, .x, .y, tune_SCR)) %>%
  mutate(sigma_num = readr::parse_number(sigma), feature_set = "SCREENED")

noise_perf_tbl <- bind_rows(noise_perf_full, noise_perf_scr)

save_tbl(noise_perf_tbl, "Table2_noise_sensitivity_oof_performance_FULL_vs_SCREENED")
show_tbl(noise_perf_tbl %>% arrange(feature_set, sigma_num, RMSE), "OOF Performance (FULL vs SCREENED)", n = 60)

# Baseline comparison table (sigma=0)
baseline_compare <- noise_perf_tbl %>%
  filter(sigma_num == 0) %>%
  arrange(feature_set, RMSE)

save_tbl(baseline_compare, "baseline_sigma0_compare_FULL_vs_SCREENED")
show_tbl(baseline_compare, "Baseline Comparison (sigma=0): FULL vs SCREENED", n = 50)

# ==============================================================
# 8) Core Figures (use SCREENED by default for cleaner story)
# ==============================================================

# Choose which feature set to plot as "main"
MAIN_SET <- "SCREENED"  # change to "FULL" if you want
perf_main <- noise_perf_tbl %>% filter(feature_set == MAIN_SET)

# Figure 1: Outcome distribution (baseline)
p_fig1 <- ggplot(train, aes(x = .data[[outcome]])) +
  geom_histogram(bins = 35, fill = COL$blue, color = "white", alpha = 0.95) +
  labs(title = "Figure 1. Distribution of Noise-Perturbed HDL Outcome (Train)",
       x = "LBDHDD_outcome (mg/dL)", y = "Count")
save_plot(p_fig1, "Figure1_outcome_distribution", w = 8, h = 5)

# Build model palette for MAIN_SET labels (because ENet/RF labels differ by tuning)
enet_name <- perf_main %>% filter(sigma_num == 0) %>% pull(model) %>% unique()
enet_name <- enet_name[str_detect(enet_name, "^ENet")]
if (length(enet_name) == 0) enet_name <- "ENet"

rf_name <- perf_main %>% filter(sigma_num == 0) %>% pull(model) %>% unique()
rf_name <- rf_name[str_detect(rf_name, "^RF")]
if (length(rf_name) == 0) rf_name <- "RF"

model_pal <- c(
  "OLS"   = "#636363",
  "Ridge" = COL$blue,
  "Lasso" = COL$red,
  setNames(COL$green, enet_name),
  setNames("#D95F02", rf_name),
  "XGB (best)" = "#7570B3"
)

# Figure 2: Baseline (sigma=0) performance bars (RMSE/MAE/R2) for MAIN_SET
perf0 <- perf_main %>%
  filter(sigma_num == 0) %>%
  select(model, RMSE, MAE, R2) %>%
  pivot_longer(cols = c(RMSE, MAE, R2), names_to = "metric", values_to = "value")

p_fig2 <- ggplot(perf0, aes(x = model, y = value, fill = model)) +
  geom_col(width = 0.75) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_fill_manual(values = model_pal) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1),
        legend.position = "none") +
  labs(title = glue("Figure 2. Baseline Cross-Validated Performance (sigma = 0) [{MAIN_SET}]"),
       x = NULL, y = NULL)
save_plot(p_fig2, "Figure2_baseline_oof_performance", w = 12, h = 6)

# Figure 3: Noise robustness curve (RMSE vs sigma) for MAIN_SET
p_fig3 <- ggplot(perf_main, aes(x = sigma_num, y = RMSE, color = model, group = model)) +
  geom_line(linewidth = 1.1, alpha = 0.95) +
  geom_point(size = 2, alpha = 0.95) +
  scale_color_manual(values = model_pal) +
  scale_x_continuous(breaks = sigma_grid) +
  labs(title = glue("Figure 3. RMSE Degradation Under Increasing Outcome Noise [{MAIN_SET}]"),
       x = "Additional noise level (sigma)", y = "OOF RMSE", color = "Model")
save_plot(p_fig3, "Figure3_rmse_degradation_curves", w = 10, h = 6)

# Figure 4: Noise robustness curve (MAE vs sigma) for MAIN_SET
p_fig4 <- ggplot(perf_main, aes(x = sigma_num, y = MAE, color = model, group = model)) +
  geom_line(linewidth = 1.1, alpha = 0.95) +
  geom_point(size = 2, alpha = 0.95) +
  scale_color_manual(values = model_pal) +
  scale_x_continuous(breaks = sigma_grid) +
  labs(title = glue("Figure 4. MAE Degradation Under Increasing Outcome Noise [{MAIN_SET}]"),
       x = "Additional noise level (sigma)", y = "OOF MAE", color = "Model")
save_plot(p_fig4, "Figure4_mae_degradation_curves", w = 10, h = 6)

# ==============================================================
# 9) Feature Stability Under Noise (Elastic-net; bootstrap)
#    (run on MAIN_SET design matrix for consistency)
# ==============================================================

B <- 300
stability_sigmas <- c(0, 5)

# pick tuning + X design for MAIN_SET
if (MAIN_SET == "SCREENED") {
  X_STAB <- x_train_scr
  tune_stab <- tune_SCR
} else {
  X_STAB <- x_train_full
  tune_stab <- tune_FULL
}

coef_no_intercept <- function(glmnet_fit, lambda) {
  b <- as.matrix(coef(glmnet_fit, s = lambda))
  b <- b[rownames(b) != "(Intercept)", , drop = FALSE]
  out <- as.numeric(b)
  names(out) <- rownames(b)
  out
}

stability_summary <- function(boot_mat, model_name, sigma_value) {
  sel_freq <- colMeans(boot_mat != 0)
  mean_beta <- colMeans(boot_mat)
  mean_abs  <- colMeans(abs(boot_mat))
  sd_beta   <- apply(boot_mat, 2, sd)
  
  sign_cons <- sapply(seq_along(sel_freq), function(j) {
    v <- boot_mat[, j]
    v_sel <- v[v != 0]
    if (length(v_sel) == 0) return(NA_real_)
    max(mean(v_sel > 0), mean(v_sel < 0))
  })
  
  tibble(
    sigma = sigma_value,
    model = model_name,
    feature = colnames(boot_mat),
    selection_freq = sel_freq,
    sign_consistency = sign_cons,
    mean_beta = mean_beta,
    mean_abs_beta = mean_abs,
    sd_beta = sd_beta,
    stability_score = selection_freq * mean_abs_beta
  ) %>%
    arrange(desc(stability_score), desc(selection_freq))
}

run_bootstrap_enet <- function(X, y, sigma_value, best_alpha, lambda_1se, B = 300, seed = 2026) {
  set.seed(seed + 1000 * sigma_value)
  n <- length(y)
  feat <- colnames(X)
  boot_mat <- matrix(0, nrow = B, ncol = length(feat), dimnames = list(NULL, feat))
  
  for (b in 1:B) {
    idx <- sample.int(n, replace = TRUE)
    fit <- glmnet(X[idx, , drop = FALSE], y[idx], alpha = best_alpha, standardize = FALSE)
    boot_mat[b, ] <- coef_no_intercept(fit, lambda = lambda_1se)
  }
  boot_mat
}

stab_all <- map_dfr(stability_sigmas, function(s) {
  y <- y_list[[paste0("sigma_", s)]]
  bm <- run_bootstrap_enet(
    X = X_STAB,
    y = y,
    sigma_value = s,
    best_alpha = tune_stab$glmnet$best_alpha,
    lambda_1se = tune_stab$glmnet$enet_lambda_1se,
    B = B
  )
  stability_summary(
    bm,
    model_name = glue("ENet(a={round(tune_stab$glmnet$best_alpha,2)}, lambda.1se) [{MAIN_SET}]"),
    sigma_value = s
  )
})

save_tbl(stab_all, glue("Table3_feature_stability_enet_selected_sigmas_{MAIN_SET}"))
save_rds(list(B = B, stability_sigmas = stability_sigmas, stab_all = stab_all, feature_set = MAIN_SET),
         glue("feature_stability_enet_{MAIN_SET}"))
show_tbl(stab_all %>% filter(sigma == 0) %>% slice(1:20),
         glue("Top Stable Features (ENet, sigma=0) [{MAIN_SET}]"), n = 20)

robust_feat_tbl <- stab_all %>%
  group_by(feature) %>%
  summarise(
    mean_selection_freq = mean(selection_freq, na.rm = TRUE),
    mean_abs_beta = mean(mean_abs_beta, na.rm = TRUE),
    mean_stability_score = mean(stability_score, na.rm = TRUE),
    sd_selection_freq = sd(selection_freq, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_stability_score))

save_tbl(robust_feat_tbl, glue("Table4_robust_feature_index_enet_{MAIN_SET}"))
show_tbl(robust_feat_tbl, glue("Robust Feature Index (ENet; selected sigmas) [{MAIN_SET}]"), n = 20)

# Figure 5: Top robust features (bar)
top_k <- 20
p_fig5 <- robust_feat_tbl %>%
  slice(1:top_k) %>%
  mutate(feature = forcats::fct_reorder(feature, mean_stability_score)) %>%
  ggplot(aes(x = feature, y = mean_stability_score)) +
  geom_col(fill = COL$green, alpha = 0.95) +
  coord_flip() +
  labs(title = glue("Figure 5. Robust Feature Importance (ENet; averaged across noise levels) [{MAIN_SET}]"),
       x = NULL, y = "Mean stability score")
save_plot(p_fig5, "Figure5_robust_feature_importance_enet", w = 10, h = 7)

# Optional: Scatter stability vs effect at sigma=0
p_scatter0 <- stab_all %>% filter(sigma == 0) %>%
  ggplot(aes(selection_freq, mean_abs_beta)) +
  geom_point(color = COL$red, alpha = 0.7, size = 2) +
  geom_smooth(method = "loess", se = FALSE, color = COL$gray, linewidth = 1) +
  labs(title = glue("Stability vs Effect Size (ENet; sigma = 0) [{MAIN_SET}]"),
       x = "Selection frequency", y = "Mean |beta|")
save_plot(p_scatter0, "scatter_stability_vs_effect_sigma0", w = 8, h = 5)

# ==============================================================
# 10) Poster Combined Panel (Figures 1–5)
# ==============================================================

combined_fig_1_5 <-
  (p_fig1 | p_fig2) /
  (p_fig3 | p_fig4) /
  (p_fig5)

combined_fig_1_5 <- combined_fig_1_5 +
  plot_annotation(
    title = glue("Figures 1–5: Robust Prediction Under Noise-Perturbed Outcomes (NHANES / ASASF) [{MAIN_SET}]")
  )

save_combined_plot(combined_fig_1_5, "PosterPanel_Figures1to5", w = 18, h = 14)

# ==============================================================
# 11) Final Fits on Full TRAIN (sigma=0) + BLIND TEST predictions
#     (Generate predictions for BOTH feature sets; choose best)
# ==============================================================

fit_and_predict_all_models <- function(x_train, x_test, y0, tune, tag = "FULL") {
  # Full-data fits
  ols_full <- fit_ols_fold(x_train, y0)
  
  ridge_full <- glmnet(x_train, y0, alpha = 0, standardize = FALSE)
  lasso_full <- glmnet(x_train, y0, alpha = 1, standardize = FALSE)
  enet_full  <- glmnet(x_train, y0, alpha = tune$glmnet$best_alpha, standardize = FALSE)
  
  rf_full  <- fit_rf(x_train, y0, mtry = tune$rf$best_mtry, seed = 2026)
  
  xgb_full <- fit_xgb_fold(x_train, y0, best_xgb_row = tune$xgb$best_xgb)
  
  tibble(
    row_id = seq_len(nrow(x_test)),
    pred_ols   = as.numeric(pred_ols(ols_full, x_test)),
    pred_ridge = as.numeric(pred_glmnet(ridge_full, x_test, tune$glmnet$ridge_lambda_1se)),
    pred_lasso = as.numeric(pred_glmnet(lasso_full, x_test, tune$glmnet$lasso_lambda_1se)),
    pred_enet  = as.numeric(pred_glmnet(enet_full,  x_test, tune$glmnet$enet_lambda_1se)),
    pred_rf    = as.numeric(pred_rf(rf_full, x_test)),
    pred_xgb   = as.numeric(pred_xgb(xgb_full, x_test)),
    feature_set = tag
  )
}

test_pred_full <- fit_and_predict_all_models(x_train_full, x_test_full, y0, tune_FULL, tag = "FULL")
test_pred_scr  <- fit_and_predict_all_models(x_train_scr,  x_test_scr,  y0, tune_SCR,  tag = "SCREENED")

test_predictions_all <- bind_rows(test_pred_full, test_pred_scr)

save_tbl(test_predictions_all, "blind_test_predictions_all_models_FULL_vs_SCREENED")
show_tbl(test_predictions_all %>% filter(feature_set == MAIN_SET), "Blind Test Predictions (MAIN_SET, Top 10)", n = 10)

# --- Competition submission file (pred.csv) ---
# Use XGBoost from MAIN_SET by default:
pred_submit <- test_predictions_all %>%
  filter(feature_set == MAIN_SET) %>%
  transmute(pred = pred_xgb)

write_csv(pred_submit, path(root_dir, "pred.csv"))
message("✅ pred.csv saved to: ", path(root_dir, "pred.csv"))

# ==============================================================
# 12) Save key objects + Reproducibility
# ==============================================================

save_rds(list(
  rec_prep = rec_prep,
  fold_id = fold_id,
  sigma_grid = sigma_grid,
  feature_screening = screen,
  tuning_full = list(glmnet = tune_full_glmnet, rf = tune_full_rf, xgb = tune_full_xgb),
  tuning_scr  = list(glmnet = tune_scr_glmnet,  rf = tune_scr_rf,  xgb = tune_scr_xgb),
  noise_perf_tbl = noise_perf_tbl,
  stab_all = stab_all,
  robust_feat_tbl = robust_feat_tbl,
  MAIN_SET = MAIN_SET
), "analysis_objects_UPDATED")

sink(path(root_dir, "sessionInfo.txt"))
cat("Reproducibility info\n")
cat("Date:", as.character(Sys.Date()), "\n")
cat("Seed: 2026\n")
cat("K folds:", K, "\n")
cat("Sigma grid:", paste(sigma_grid, collapse = ", "), "\n")
cat("MAIN feature set:", MAIN_SET, "\n")
cat("\nFeature screening:\n")
cat("FULL p:", ncol(x_train_full), "\n")
cat("SCREENED p:", ncol(x_train_scr), "\n")
cat("Correlation cutoff:", 0.90, "\n\n")
cat("Bootstrap B:", B, "\n")
cat("Stability sigmas:", paste(stability_sigmas, collapse = ", "), "\n\n")
print(sessionInfo())
sink()

message("Done. Outputs saved to: ", root_dir)
message("Tables:   ", path(root_dir, "tables"))
message("Figures:  ", path(root_dir, "figures"))
message("RDS:      ", path(root_dir, "rds"))
message("Blind test predictions saved: tables/blind_test_predictions_all_models_FULL_vs_SCREENED.csv")
message("Submission file saved: pred.csv")
