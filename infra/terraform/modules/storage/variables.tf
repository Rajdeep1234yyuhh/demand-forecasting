variable "compartment_id" {
  description = "OCID of the compartment for the buckets."
  type        = string
}

variable "region" {
  description = "OCI region; used to grant the Object Storage service lifecycle permissions."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for bucket names."
  type        = string
}

variable "tags" {
  description = "Freeform tags applied to every bucket."
  type        = map(string)
  default     = {}
}

variable "buckets" {
  description = "Buckets to create, keyed by purpose."
  type = map(object({
    noncurrent_version_retention_days = number
    object_retention_days             = optional(number)
  }))
}
