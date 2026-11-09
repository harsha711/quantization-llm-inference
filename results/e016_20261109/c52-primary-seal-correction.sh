# Reconstructed equivalent of the manifest correction, not an executed script.
# Do not run against the ended allocation. Original failed script is preserved separately.
set -euo pipefail
R=/mnt/beegfs/hpuvvad/csc548-w4a16-e016-final-20261109/results/e016_20261109
cp "$R/primary-retrieval.sha256" "$R/primary-retrieval-self-reference.sha256"
find "$R" -maxdepth 1 -type f -name 'primary*' ! -name 'primary-retrieval.sha256' -print0 | sort -z | xargs -0 sha256sum > "$R/primary-retrieval.sha256"
