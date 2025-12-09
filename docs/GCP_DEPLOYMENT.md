\# GCP Deployment Guide - PitchQuest



\## Deployed By

Harshita Shitut



\## Live URLs

\- \*\*Frontend:\*\* http://136.116.237.83:3000

\- \*\*Backend API:\*\* http://136.116.237.83:8000/docs



\## GCP Configuration



\### VM Specifications

\- \*\*Project:\*\* mlops-pitchquest

\- \*\*Instance:\*\* pitchquest-server

\- \*\*Machine Type:\*\* e2-medium (2 vCPU, 4 GB RAM)

\- \*\*OS:\*\* Ubuntu 22.04 LTS Minimal

\- \*\*Disk:\*\* 30 GB

\- \*\*Region:\*\* us-central1-a



\### Firewall Rules

| Port | Service |

|------|---------|

| 80 | HTTP |

| 3000 | Frontend |

| 8000 | Backend API |

| 5432 | PostgreSQL |

| 6379 | Redis |



\## Deployment Steps

1\. Create GCP VM with Ubuntu 22.04

2\. Install Docker and Docker Compose

3\. Clone repository

4\. Create .env file with API keys

5\. Run `docker-compose up --build`



\## Services Running

\- Frontend (React/Vite)

\- Backend (FastAPI)

\- PostgreSQL (via Supabase)

\- Redis (caching)

