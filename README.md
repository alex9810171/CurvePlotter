## How to use
```
sh plot.sh
```
- You can edit `plot.sh` to fit your usage.
- we use folder name to give color

## Algorithm
- Use sigmoid function when over 3 points.
- Use log function when under and include 3 points or x_max-x_min not over 2/3*x_max.

## Library
```
matplotlib
scipy
pandas
```

## Allow axis list
```
r23_package_power
r23_point
```

## Make sure avoid
- no over 20 points to fit for each curve
- should not include "strange points" to fit