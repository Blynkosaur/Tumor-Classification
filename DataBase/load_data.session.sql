SHOW databases

-- @block
create table BC_DATA (
    id int
)
-- @block
LOAD DATA INFILE '../data/breast-cancer.csv' into table BC_DATA
FIELDS TERMINATED BY ','
IGNORE 1 LINES;

SELECT * FROM BC_DATA
-- @block
SHOW VARIABLES LIKE 'secure_file_priv';
--@block
SELECT * from BC_DATA
--@block
UPDATE mysql.user SET File_priv = 'Y' WHERE Host = 'localhost' AND User = 'root';
FLUSH PRIVILEGES;
--@block

SELECT * from layerone_weights
--@block
INSERT INTO layerone_weights(weight_1) VALUES (0.01);
--@block
INSERT INTO layerthree_bias (weight_1) VALUES (0.2866344649543016);
