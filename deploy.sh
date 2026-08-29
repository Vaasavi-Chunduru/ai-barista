#!/usr/bin/env bash
# Deploys the AI Barista agent to Cloud Run.
#
# Usage:
#   export PROJECT_ID="your-gcp-project"
#   export REGION="us-central1"
#   ./deploy.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID env var first}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="ai-barista"
SA_NAME="ai-barista-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"

echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# --- Least-privilege service account -------------------------------------
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  echo "Creating service account ${SA_EMAIL}..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="AI Barista runtime service account"
fi

echo "Granting only the Vertex AI user role (least privilege)..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None

# --- Deploy ----------------------------------------------------------------
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --service-account "${SA_EMAIL}" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION},GOOGLE_GENAI_USE_VERTEXAI=True" \
  --allow-unauthenticated \
  --min-instances 0

echo "Done. Service URL:"
gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)'
