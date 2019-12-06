Document REST API

The virtual environment has been bundled into the folder
To run the app you will need to run the following commands - 
1 - source venv/bin/activate
2 - python app.py
3 - OPTIONAL - python app1.py
Running 1 activates the virtual environment
Running 2 activates the server which runs the REST service

In case you want to enter the plot yourself by typing it you can run 3
This gives you an index page to type in the plot and returns the same JSON
Additionally, you can get the same API functionality from running app1.py


The REST api gives you back a JSON dictionary in the following format - 
{"labels":[label1,label2..],"status":"success"}

You can fetch the labels of the current plot by calling "labels" in the JSON data returned
