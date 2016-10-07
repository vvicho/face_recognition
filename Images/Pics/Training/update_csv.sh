filename="data.csv"

# if file already exists, remove it
if [ -f $filename ] ; then
  echo 'Updating CSV file.'
  rm "$filename"
fi

# For every directory
for d in */ ; do
  # Get the name of the person with the slash character trimmed
  person=$(echo $d | tr -d '/')

  # Go into that persons folder
  cd "$d"

  # Get every photo in the folder
  for entry in * ; do
     # Get number and name of the folder
     number=$(echo $person |cut -d'.' -f1)
     name=$(echo $person |cut -d'.' -f2)
     # Save the line in a CSV file
     line="$PWD/$entry,$number,$name"
     echo "$line" >> ../$filename
  done

  # Go back to the main folder to continue with the next one
  cd ..  
done

echo 'CSV file updated.'
