terraform {
  required_version = ">= 1.6"

  required_providers {
    oci = {
      source  = "oracle/oci"
      version = ">= 6.0, < 8.0"
    }
  }

  # State lives in OCI Object Storage through its S3-compatible API.
  # Supply settings with: terraform init -backend-config=backend.hcl
  backend "s3" {}
}
