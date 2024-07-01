# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:13:15 2021

@author: ddiaz
"""


#%%

import pandas as pd
import numpy as np
import datetime
from faker import Faker



# function to create a dataframe with fake values for our workers
def make_event_data(num, day_number, fake):
    '''
    Makes event data for a user

    Parameters
    ----------
    num : int with number of events (queries) that want to be added for each
    user

    Returns
    -------
    fake_events: TYPE
        DESCRIPTION.

    '''
    device_type = ['Phone', 'Notebook', 'Smart Speaker', 'Smart TV', 'Other']
    phone_model = ['Pixel', 'Samsung Galaxy', 'iPhone 12', 'LG V30', 'Other']
    os = ['Android 11', 'Android 10', 'Windows 10', 'iOS', 'Other']
    browser = ['Safari', 'Chrome', 'Edge', 'Explorer', 'Other']
    query_class = ['HEAD', 'TORSO', 'TAIL']

    fake_events = [{'query_or_click':np.random.choice([0, 1], p=[0.05,0.95]),
                  'shown_onebox':np.random.choice([0, 1], p=[0.7,0.3]),
                  'shown_spell_correction':np.random.choice([0, 1]),
                  'shown_another_property':np.random.choice([0, 1]),
                  'shown_onebox_with_map':np.random.choice([0, 1]),
                  #'shown_onebox_without_map':np.random.choice([0, 1]),
                  'shown_organic_result':np.random.choice([0, 1]),
                  'shown_top_text_ad':np.random.choice([0, 1]),
                  'shown_pla':np.random.choice([0, 1]),
                  'shown_rhs_ad':np.random.choice([0, 1]),
                  'timestamp_seconds': fake_date_for_day(day_number, fake),
                  'device_type':np.random.choice(device_type,\
                                                 p=[0.50, 0.30, 0.10, 0.05,\
                                                    0.05]),
                  'query_class': np.random.choice(query_class, p=[0.1, 0.8,
                                                                  0.1]),
                  'phone_model':np.random.choice(phone_model,\
                                                 p=[0.05, 0.2, 0.4, 0.1,\
                                                    0.25]),
                  'operating_system':np.random.choice(os), #we set equal probabilities for OS
                  'browser':np.random.choice(browser, p=[0.30, 0.50, 0.05,\
                                                         0.05, 0.1])}\
                    for x in range(num)]
        #note that data is not validated yet. For example, only an event that is either
        #a query or a click should be able to have True values for the rest
    return fake_events



def fake_date_for_day(day_number, fake):

    if day_number == 1:
        start_date = datetime.datetime(2019, 10, 22, 0, 0, 0)
        end_date = datetime.datetime(2019, 10, 22, 23, 59, 59)

    elif day_number == 2:
        start_date = datetime.datetime(2019, 10, 24, 0, 0, 0)
        end_date = datetime.datetime(2019, 10, 24, 23, 59, 59)

    elif day_number == 3:
        start_date = datetime.datetime(2021, 5, 4, 0, 0, 0)
        end_date = datetime.datetime(2021, 5, 4, 23, 59, 59)

    else:
        start_date = datetime.datetime(2021, 5, 6, 0, 0, 0)
        end_date = datetime.datetime(2021, 5, 6, 23, 59, 59)

    return fake.date_time_ad(start_datetime = start_date, end_datetime = \
                             end_date)


# function to create a dataframe with fake values for the users
def make_sessions(n_users=100):

    fake_session_ids = [{'session_id':np.random.randint(10000000, 100000000-1)}\
                    for x in range(n_users)]

    return fake_session_ids



def make_fake_table_1(n_users=100):
    # empty list to store our events dataframes in
    dfs_list = []
    session_df = pd.DataFrame(make_sessions(n_users))
    fake = Faker()
    for index, row in session_df.iterrows():

        # make events for each session id
        # note that the number of queries we are getting from a random
        # distribution, truncated in 0
        n_queries = np.random.normal(50, 30, 1).round().astype(int)[0]
        if n_queries < 0:
            n_queries = 0

        day_number = np.random.choice([1,2,3,4])
        events = pd.DataFrame(make_event_data(n_queries, day_number, fake))

        # add session id so we know who made the queries/clicks
        events['session_id'] = row['session_id']

        # append to df list
        dfs_list.append(events)

    # concatenate all the dfs
    df = pd.concat(dfs_list)
    # clean for phone model, makes sure is available only for phones
    df.loc[df['device_type'] != 'Phone', 'phone_model'] = 'NA'
    return df



#%%

# function to create a dataframe with fake values for our workers
def make_query_table(df_events):
    '''
    Makes event data for a user

    Parameters
    ----------
    num : int with number of events (queries) that want to be added for the
    user

    Returns
    -------
    fake_events: TYPE
        DESCRIPTION.

    '''
    fake = Faker()
    events = list(df_events['event_id'])
    query_df = pd.DataFrame()
    for event_id in events:
        fake_query = make_fake_query(fake, event_id)
        query_df = query_df.append(fake_query, ignore_index=True)
    return query_df

def make_fake_query(fake, event_id):

    vertical4 = ['Arts', 'Vehicles', 'Sports', 'Shopping', 'Hobbies', 'Other']
    query_intent= ['Shopping', 'Search', 'Navigational', 'Other']

    fake_query = {'query_id':event_id,
                  'is_image_q':np.random.choice([0, 1], p=[0.80, 0.2]),
                  'is_map_q':np.random.choice([0, 1], p=[0.95, 0.05]),
                  'is_navigational':np.random.choice([0, 1]),
                  'is_commercial':np.random.choice([0, 1]),
                  'vertical4':np.random.choice(vertical4),
                  'query_intent':np.random.choice(query_intent),
                  'abandoned':np.random.choice([0, 1])}
        #note that data is not validated yet. For example, only an event that is either
        #a query or a click should be able to have True values for the rest
    return fake_query



#%%

# function to create a dataframe with fake values for our workers
def make_click_table(df_events):
    '''
    Makes event data for a user

    Parameters
    ----------
    num : int with number of events (queries) that want to be added for the
    user

    Returns
    -------
    fake_events: TYPE
        DESCRIPTION.

    '''
    fake = Faker()
    events = list(df_events['event_id'])
    click_df = pd.DataFrame()
    for event_id in events:
        fake_click = make_fake_click(fake, event_id)
        click_df = click_df.append(fake_click, ignore_index=True)
    return click_df

def make_fake_click(fake, event_id):

    fake_click = {'click_id':event_id,
                  'is_good':np.random.choice([0, 1]),
                  'is_organic':np.random.choice([0, 1]),
                  'is_top_text_ad':np.random.choice([0, 1]),
                  'is_web_link':np.random.choice([0, 1]),
                  'next_page':np.random.choice([0, 1]),
                  'is_rhs_ad':np.random.choice([0, 1]),
                  'is_onebox':np.random.choice([0, 1]),
                  'is_spell_correction':np.random.choice([0, 1])}
        #note that data is not validated yet. For example, only an event that is either
        #a query or a click should be able to have True values for the rest
    return fake_click



#%%


def main():
    '''
    Creates the mock data and saves into pickle and csv formats
    We set the number of sessions in the data to 200 but it can be easily
    changed
    We set the event_id as the index to make sure they don't repeat and add a
    fixed quantity to maintain the number of digits
    We split the event data that is either query or click in half and make click
    data with one half and query data with the other half

    We implement more realism with the onebox. Now withmap and without map
    are mutually exclusive
    Returns
    -------
    None.

    '''
    table_1 = make_fake_table_1(200).reset_index()
    table_1 = table_1.drop('index', axis=1)
    table_1.loc[:, 'shown_onebox_without_map'] = abs(table_1['shown_onebox_with_map'] - 1)
    table_1.loc[:,'shown_onebox_without_map'] = table_1['shown_onebox_without_map']*\
        table_1['shown_onebox']
    table_1.loc[:,'shown_onebox_with_map'] = table_1['shown_onebox_with_map']*\
        table_1['shown_onebox']
    table_1['event_id'] = table_1.index + 10000
    query_or_click_events = table_1[table_1['query_or_click'] == 1]
    sample_50 = query_or_click_events.sample(frac = 0.5)
    rest_50 = query_or_click_events.drop(sample_50.index)
    table_2 = make_query_table(sample_50)
    table_3 = make_click_table(rest_50)
    table_1.to_pickle('../output/event_table.pkl')
    table_2.to_pickle('../output/query_table.pkl')
    table_3.to_pickle('../output/click_table.pkl')

    table_1.to_csv('../output/event_table.csv')
    table_2.to_csv('../output/query_table.csv')
    table_3.to_csv('../output/click_table.csv')



