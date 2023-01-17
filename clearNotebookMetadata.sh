# Clears kernel spec all example notebooks
for file in examples/**/*.ipynb
do
   jq 'del(.metadata.kernelspec)' "$file" | sponge "$file"
done
