variable "region" {
  description = "OCI home region. Always Free compute is only available here."
  type        = string
}

variable "oci_profile" {
  description = "Profile name in ~/.oci/config."
  type        = string
  default     = "DEFAULT"
}

variable "compartment_id" {
  description = "OCID of the compartment that holds all project resources."
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names."
  type        = string
  default     = "df"
}
