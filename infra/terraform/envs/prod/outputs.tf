output "bucket_names" {
  description = "Object Storage bucket names, keyed by purpose."
  value       = module.storage.bucket_names
}

output "object_storage_namespace" {
  description = "Tenancy Object Storage namespace."
  value       = module.storage.namespace
}
