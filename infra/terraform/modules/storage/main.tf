data "oci_objectstorage_namespace" "this" {
  compartment_id = var.compartment_id
}

# Lifecycle rules only run if the Object Storage service may manage objects here.
resource "oci_identity_policy" "lifecycle" {
  compartment_id = var.compartment_id
  name           = "${var.name_prefix}-objectstorage-lifecycle"
  description    = "Allow Object Storage lifecycle policies to delete expired objects."
  statements = [
    "Allow service objectstorage-${var.region} to manage object-family in compartment id ${var.compartment_id}",
  ]
  freeform_tags = var.tags
}

resource "oci_objectstorage_bucket" "this" {
  for_each = var.buckets

  compartment_id = var.compartment_id
  namespace      = data.oci_objectstorage_namespace.this.namespace
  name           = "${var.name_prefix}-${each.key}"
  access_type    = "NoPublicAccess"
  storage_tier   = "Standard"
  versioning     = "Enabled"
  freeform_tags  = var.tags
}

resource "oci_objectstorage_object_lifecycle_policy" "this" {
  for_each = var.buckets

  namespace = data.oci_objectstorage_namespace.this.namespace
  bucket    = oci_objectstorage_bucket.this[each.key].name

  rules {
    name        = "expire-previous-versions"
    action      = "DELETE"
    target      = "previous-object-versions"
    time_amount = each.value.noncurrent_version_retention_days
    time_unit   = "DAYS"
    is_enabled  = true
  }

  dynamic "rules" {
    for_each = each.value.object_retention_days == null ? [] : [each.value.object_retention_days]
    content {
      name        = "expire-objects"
      action      = "DELETE"
      target      = "objects"
      time_amount = rules.value
      time_unit   = "DAYS"
      is_enabled  = true
    }
  }

  depends_on = [oci_identity_policy.lifecycle]
}
