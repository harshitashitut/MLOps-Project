output "vm_external_ip" {
  description = "External IP of the VM"
  value       = google_compute_address.pitchquest_ip.address
}

output "frontend_url" {
  description = "Frontend URL"
  value       = "http://${google_compute_address.pitchquest_ip.address}:3000"
}

output "backend_url" {
  description = "Backend API URL"
  value       = "http://${google_compute_address.pitchquest_ip.address}:8000/docs"
}