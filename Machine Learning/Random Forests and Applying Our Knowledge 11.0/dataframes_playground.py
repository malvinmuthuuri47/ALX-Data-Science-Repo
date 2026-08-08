"""
This module contains code that's meant to introduce the learner to filtering a dataframe using data from
another dataframe
"""

import pandas as pd

students = pd.DataFrame({
    'StudentID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Faith', 'George', 'Helen'],
    'Department': ['Computer Science', 'Mathematics', 'Computer Science', 'Physics', 'Mathematics', 'Computer Science', 'Physics', 'Computer Science'],
    'Year': [1, 2, 3, 1, 4, 2, 3, 4]
}).set_index('Department')

# print(students)

scholarships = pd.DataFrame({
    'Department': [
        'Computer Science', 'Physics'
    ],
    'Scholarship': [
        'AI Excellence', 'Research Grant'
    ]
}).set_index('Department')

# print(scholarships)

'''Step One: Extract departments from the second dataframe'''
eligible_departments = scholarships.index

# print(eligible_departments)

'''Filter students who are in select departments'''
eligible_students = students.loc[students.index.isin(eligible_departments)]

print(eligible_students)