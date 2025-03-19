# Author: Philipp Kleer
# E-Mail: philipp.kleer@posteo.com

# General approach

I wanted to have as few as possible functions, therefore, I created a function `generate_report()` that takes the metrics as argument to apply this function to all cases. Hereby, I can use this function in all cases and do not need to differentiate the functions across the cases. 

# Questions

## After step 4, explain what changed over the weeks 1, 2, and 3?

- we can see that the mean of the target `cnt` is way more seasonal in these three weeks than it is in the reference data. It has higher peaks,too. Probably, there were some warmer days in February (we see higher correlation to `temp`, and `hum`, for example).

## After step 5, explain what seems to be the root cause of the drift (only using data)?

- we see a data drift in: `temp`, `windspeed`, `hum`, and `atemp`. 
- therefore, weather conditions are the root cause of the drift. 

## After step 6, explain what strategy to apply?

- retraining model with drifted data (or with more months than just january)
- assure maybe last 12 months in training data to better monitor these seasonal trends that might occur in winter (sunny warm days) or in summer (cold, rainy days)
- a model for weather data with just a single winter months as reference might not be a good reference model/data.

# Commands to run it
```bash
python exam_evidently.py

evidently ui --workspace ./exam_evidently/
```
