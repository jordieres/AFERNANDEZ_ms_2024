#!/bin/bash 
# Verifica si se pasaron exactamente dos argumentos
if [ "$#" -ne 2 ]; then
  echo "Uso: $0 'ID_declase' '<patrón_de_archivos>'"
  exit 1
fi
a=`ls $2`
# echo $2
for file in $a; do
  if [ -f "$file" ]; then
    # Extraer el nombre del archivo sin la ruta
    base_name=$(basename "$file")
    dir_name=$(dirname "$file")

    # Sustituir todo hasta el primer '+' por 'clase_0'
    new_name=$(echo "$base_name" | sed "s/^[^+]*+/clase_$1+/")
    newf=$(echo "$base_name" | sed 's/^[^+]*+/clase_*+/')
    exis=`find "$dir_name" -maxdepth 1 -name "$newf" | wc -l`
    if [ $exis -eq 0 ]; then
       cp "$dir_name"/"$base_name" "$dir_name"/"$new_name"
    fi
  fi
done
