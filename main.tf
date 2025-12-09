terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_address" "pitchquest_ip" {
  name = "pitchquest-static-ip"
}

resource "google_compute_firewall" "pitchquest_firewall" {
  name    = "pitchquest-allow-ports"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "3000", "8000", "5432", "6379"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["pitchquest-server"]
}

resource "google_compute_instance" "pitchquest_vm" {
  name         = "pitchquest-server"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["pitchquest-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.pitchquest_ip.address
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y docker.io docker-compose git
    usermod -aG docker $USER
    systemctl enable docker
    systemctl start docker
  EOF
}