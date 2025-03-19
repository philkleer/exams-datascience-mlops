# Airflow Exam

# Name: Philipp Kleer

# E-Mail: philipp.kleer@posteo.com

# Approach

For my approach, I chose to follow the decorator's style since I think it is the most readable. Furthermore, I splitted up the script to be separated in helper functions (first part), then functions that are tasks (second part) and then the definition of the DAG (third part). 

Since with decorators, Airflow directly manage task instances, I could simplify the xcom-savings by simple using `return`. Compared to the "standard" way, this facilitates the code and its readability a lot. Even with this, I can directly call the value in other functions. 

I grouped the verification tasks after creating the data sets in `verification_group` and I also grouped (as suggested) the subtasks of task 4 in `model_group`.

# Problems I encountered

1. verification task: I did a wrong call on the file_sensor so that it never worked
2. load_data: I forgot to serialize X and y before returning it into the xcom, therefore, I got an error. I fixed this and created csv_strings that are serializable. I added a reformatting function into the task 4 functions to reformat to a pd.DataFrame (feature matrix) and pd.Series (target).
3. I struggled a bit with the xcom data and loading it for the next task: however, my biggest mistake was to oversee that the taskid is not `load_task4`, it was `load_data`. 
4. 15 minutes run: I wanted to handle this automatically which was a bit tricky, since I only want to wait for the first 20 minutes. I created a timestamp variable that checks if 20 minutes are already passed and only then, it goes further. Hereby, I secured that the delay is only given for the first runs and not for the later ones that have already enough data. I got first an error, because the variables are saved as string, therefore, I needed to convert it when reloaded to make the substraction happening. Furthermore, I ran into trouble with the `BranchPythonOperator`, but then found a solution with the `ShortCircuitOperator`.