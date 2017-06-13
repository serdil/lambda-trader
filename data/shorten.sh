

for file in orig/*csv
do
    SIZE=2000
    LINES=$(cat $file | wc -l)
    END=$((LINES-$SIZE))
    OUTFILE=${file:5}

    echo $OUTFILE
    if (( $END > 2 ))
    then
        cat $file | sed '2,'"$END"'d' > "$OUTFILE"
    else
        cat $file > "$OUTFILE"
    fi
done
