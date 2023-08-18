# Installing Tensorflow GPU version
FROM python:3.9.17
# set dir

# RUN pip3 install -r requirements.txt

# Running installation for python
# RUN pip install pytorch==1.12.0
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install scipy
COPY . .
# RUN python3 hello.py
# CMD ["python3", "hello.py"]