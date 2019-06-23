# -*- coding: utf-8 -*-
"""
need to define two database, one for local, one for remote
"""
from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime, BigInteger, SmallInteger

metadata=MetaData()
mtrp_alarm=Table('mtrp_alarm',metadata,
                 Column('id',BigInteger(),primary_key=True),
                 Column('isDel',Integer()),
                 Column('alarmTime',DateTime),
                 Column('confirmTime',DateTime),
                 Column('event_id',Integer()),
                 Column('content',String(255)),
                 Column('createTime',DateTime),
                 Column('device_id',BigInteger()),
                 Column('channel_type',SmallInteger()),
                 Column('channel_no',SmallInteger()),
                 Column('feedback',String(255)),
                 Column('feedbackTime',DateTime),
                 Column('pic_name',String(255)),
                 Column('pic_url',String(255)),
                 Column('smsTime',DateTime),
                 Column('status',Integer()),
                 Column('valid',Integer()),
                 Column('fileUrl',String(255)),
                 Column('logID',Integer()),
                 Column('alarmLevel_id',BigInteger()),
                 Column('ip',String(255))
                 )

mtrp_alarm_type=Table('mtrp_alarm',metadata,
                      Column('id',Integer(),primary_key=True),
                      Column('event_name',String(50)),
                      Column('event_priority',SmallInteger()),
                      Column('event_type',SmallInteger()),
                      Column('event_sub_type1',SmallInteger()),
                      Column('event_sub_type2',SmallInteger()),
                      Column('event_level',SmallInteger()),
                      Column('syntax',String(255)),
                      )