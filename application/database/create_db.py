# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
# import os


# if not os.path.exists('db.sqlite'):
#     engine = create_engine("sqlite:///db.sqlite")
#     engine.execute("CREATE DATABASE db.sqlite")
#     print("The database file has been created.")
# else:
#     print("The database file already exists.")


# Base = declarative_base()

# class Prediction(Base):
#     __tablename__ = 'predictions'

#     id = Column(Integer, primary_key=True)
#     text = Column(String)
#     predicted_type = Column(String)

# engine = create_engine('sqlite:///predictions.db')
# Base.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)
