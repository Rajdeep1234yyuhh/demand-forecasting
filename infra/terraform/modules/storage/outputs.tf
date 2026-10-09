output "bucket_names" {
  description = "Bucket names keyed by purpose."
  value       = { for key, bucket in oci_objectstorage_bucket.this : key => bucket.name }
}

output "namespace" {
  description = "Object Storage namespace."
  value       = data.oci_objectstorage_namespace.this.namespace
}
