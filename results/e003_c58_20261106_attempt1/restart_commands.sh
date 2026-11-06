set -euo pipefail
cd /home/hpuvvad/csc548-w4a16-e003-20261106
old=results/e003_c58_20261106_attempt1
new=results/e003_c58_20261106
[[ ! -e "$old" ]]
mv "$new" "$old"
mkdir "$new"
cp "$old"/{allocation.txt,launcher.log,build.log,smoke.log,check.log,check.sha256,prepare_commands.sh,e002_saved.sass,e002_rebuilt.sass,naive_saved.sass,naive_rebuilt.sass,e003.sass} "$new/"
cp /tmp/e003_measure_commands.sh "$new/measure_commands.sh"
sha256sum -c "$new/check.sha256"
bash "$new/measure_commands.sh"
