# AutoML
luigi workflow for basic ML processing

to run this workflow:
PYTHONPATH='.' luigi --module postprocess_to_out_cv postprocess_cv --is-local True --reg 'XGB' --csv-name 'output' \
--list-cv '[{"md": 3}, {"md": 6}]' --workers 2

![alt tag](https://github.com/AminKhribi/AutoML/blob/master/luigi_cv.png)


