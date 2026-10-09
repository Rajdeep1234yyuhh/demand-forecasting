locals {
  tags = {
    project    = "demand-forecasting"
    managed_by = "terraform"
  }
}

module "storage" {
  source = "../../modules/storage"

  compartment_id = var.compartment_id
  region         = var.region
  name_prefix    = var.name_prefix
  tags           = local.tags

  buckets = {
    # MLflow artifacts, model files, raw and processed data.
    artifacts = {
      noncurrent_version_retention_days = 30
    }
    # Nightly pg_dump backups of the Neon database.
    backups = {
      noncurrent_version_retention_days = 7
      object_retention_days             = 30
    }
  }
}

# Network and cluster modules land in Phase 7.
