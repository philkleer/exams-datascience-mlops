#!/bin/cat

echo "1. Statement of question 1"
cat people.json | jq .[] | jq '{name: .name, attributes_count: (length)}' | head -n 12
echo "Command: From the json file we take the name put it into 'name'. Furthermore we create an attributes_count (inclluding all attributes) and then we show the first 12 lines."
echo "Answer: we see three persons and each of them has 16 attributes; therefore, it might be true that all documents have 17 attributes. The three first persons (documents) are named Luke Skywalker, C-3PO and R2-D2"
echo -e "\n--------------------------------------\n"

echo "2. Statement of question 2"
cat people.json | jq .[] | jq 'select(.birth_year == "unknown") | length' | tail -n 1
echo "Command: We take the json file to jq, use select to get the attribute .birth_year and filter where it is 'unknown'. Then we pip this and take the length. Lastly we get the tail of it"
echo "Answer: There are 17 unknown values in birth_year"
echo -e "\n--------------------------------------\n"

echo "3. Statement of question 3"
cat people.json | jq -r '.[] | "\(.created | split("T")[0] | split("-")[0])-\(.created | split("T")[0] | split("-")[1])-\(.created | split("T")[0] | split("-")[2]): \(.name)"' | head -n 10
echo "Command: We take the json file and pip it to jq. then we use the -r option to provide raw output. We take the variable created, split it three times for year, month, and day. and then we add the name. Lastly we use head to display only 10 entries."
echo "Answer: don't need an extra answer"
echo -e "\n--------------------------------------\n"

echo "4. Statement of question 4"
jq -r 'group_by(.birth_year)[] | select(length > 1) | map(.id) | . as $ids | if length > 1 then [[$ids[0], $ids[1]]] else [] end' people.json
echo "Command: I Used directly jq and called the object last. Again I use -r to get raw output. I group by birth_year, pipe it and select the cases where are more than 1 (double birthday). then I pip it again and then give an id to each element. I pipe this and save the each id in ids. I pipe it again and if it is greater then one, I print it out to show the id's that have the same birthday" 
echo "Answer: There are 7 pairs of characters that are born at the same time. I.e., the first pair are the characters with id's 4 and 11, the second 6 and 36, and o on."
echo -e "\n--------------------------------------\n"

echo "5. Statement of question 5"
jq -r '.[] | "\(.films[0]): \(.name)"' people.json | head -n 10
echo "Command: I take a raw output again and just make a string from first entry in films and the name. I display only 10 cases."
echo "Answer: Don't need an extra answer"
echo -e "\n----------------BONUS----------------\n"

echo "6. Statement of question 6"
jq 'map(select((.height | tonumber? // empty) != null))' people.json > filtered_people.json
echo "Command: We use map to apply the expression to each element. We filter with select based on the condition. The condition tries to get height to number, if that doesn't work (it's not a number!) it returns empty. Therefore, we an set the condition to select cases that ar not null. we do this with the data people.json and save it into filtered_people.json "
echo "Answer: Don't need an extra answer"
echo -e "\n--------------------------------------\n"

echo "7. Statement of question 7"
jq 'map(.height |= tonumber)' filtered_people.json > filtered2_people.json
echo "Command: Similar logic as above. We use map to apply to each element. We transform all hieghts to a number. Remember we filtered values that cannot be transofrmed to numbers before. We save the new dataset to filtered2_people.json"
echo "Answer: don't need an extra answer"
echo -e "\n--------------------------------------\n"

echo "8. Statement of question 8"
jq 'map(select(.height >= 156 and .height <= 171))' filtered2_people.json
echo "Command: We simply apply a select function depending on the height and plot the filtered data set."
echo "Answer: don't need an extra answer"
echo -e "\n--------------------------------------\n"

echo "9. Statement of question 9"
jq -r 'min_by(.height) | "\(.name) is \(.height) tall"' people.json > people_9.txt
echo "Command: Again we use raw output. We take the minimum value of people.json and then make a string that we save into people_9.txt."
echo "Answer: don't need an extra answer"
echo -e "\n--------------------------------------\n"
