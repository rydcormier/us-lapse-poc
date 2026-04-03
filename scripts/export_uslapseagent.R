#!/usr/bin/env Rscript

# Script to export the uslapseagent dataset from CASdatasets
# to a specified output file in CSV or Parquet format.

args <- commandArgs(trailingOnly = TRUE)
out <- "data/raw/uslapseagent.csv"

if (length(args) >= 2 && args[1] == "--out") {
    out <- args[2]
}

message("Output path: ", out)

if (!requireNamespace("CASdatasets", quietly = TRUE)) {
    stop(
        "CASdatasets not installed.\n",
        "Install from https://cas.uqam.ca/pub/
        (source) or GitHub dutangc/CASdatasets.\n",
        "\n",
        "If GitHub install fails with 'there is no package called \'ps\'', run:\n",
        "  install.packages(c('ps','processx','remotes'))\n",
        "  remotes::install_github('dutangc/CASdatasets')\n",
        "Then re-run this script."
    )
}

suppressPackageStartupMessages(library(CASdatasets))

data(uslapseagent, package = "CASdatasets")

df <- uslapseagent
df$policy_id <- seq_len(nrow(df))
df <- df[, c("policy_id", setdiff(names(df), "policy_id"))]

dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)

# If writing parquet and arrow is available, use parquet; otherwise fall back to CSV.

if (grepl("\\.parquet$", out) && requireNamespace("arrow", quietly = TRUE)) {
    suppressPackageStartupMessages(library(arrow))
    arrow::write_parquet(df, out)
    message("Wrote parquet: ", out)
} else {
    write.csv(df, out, row.names = FALSE)
    message("Wrote csv: ", out)
}