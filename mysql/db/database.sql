
CREATE DATABASE IF NOT EXISTS database;

USE database;

CREATE TABLE user(
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(256) NOT NULL,
    first_name VARCHAR(256) NOT NULL,
    last_name VARCHAR(256) NOT NULL,
    password_hash CHAR(60) NOT NULL,
    role ENUM(admin, standard) NOT NULL,
    PRIMARY KEY (id),
    UNIQUE KEY email (email)
);

CREATE TABLE prediction(
    id INT NOT NULL AUTO_INCREMENT,
    input_text TEXT NOT NULL,
    predicted_type VARCHAR(256) NOT NULL,
    user_id INT,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id),
    FOREIGN KEY (user_id) REFERENCES user(id)
);

INSERT INTO user (email, first_name, last_name, password_hash, role)
VALUES
('Neo', 'Thomas', 'Anderson', '$2b$12$iOZQSHDfAUUE73PKFMzpEOkxyqgeKcAGtPFWQXk04fVMFarYBL1aC', 'admin'),
('Arnold', 'Arnold', 'T1000', '$2b$12$OS3rgLnyPieDX0bcbbO1vOwJfz0TMotbSDPR3IfoCIcxzfrZm/h4S', 'standard'),
('Sarah', 'Sarah', 'Canard' '$2b$12$lZVgsZANfnA78cyN0PaAxeMWCJtXcNK8sCQ.O1OLtSzI35Le0XgK2', 'standard'),
('John', 'John', 'Canard', '$2b$12$Ue2FU.RcQQvGQf61TwTGh.veFSngJPdIRQcqnvmDddHisTTYX9I9.', 'standard');