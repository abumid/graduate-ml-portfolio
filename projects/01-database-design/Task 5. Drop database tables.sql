/*DRPT Coursework by 001516, 001631, 001805*/
USE KFCDB
GO

--Task 2.5 Deletion of all the tables
DROP TABLE Supply
GO		   
DROP TABLE Feedback
GO		
DROP TABLE EmployeeSchedule
GO
DROP TABLE SupplierContactPerson
GO
DROP TABLE Supplier
GO	
DROP TABLE OrderDetail
GO
DROP TABLE [Transaction]
GO
DROP TABLE [Order]
GO
-- In our database we have cross reference in Employee and Restaurant tables. 
-- Thus we have to drop foreign key constraints first to be able to drop tables.
ALTER TABLE Employee DROP CONSTRAINT FK_Employee_Restaurant
GO
ALTER TABLE Restaurant DROP CONSTRAINT FK_Restaurant_Manager
GO
DROP TABLE Employee
GO	
DROP TABLE Restaurant
GO	  
DROP TABLE PaymentMethod
GO		
DROP TABLE Customer
GO
DROP TABLE Product
GO	
DROP TABLE Position
GO	
DROP TABLE FoodCategory
GO	
DROP TABLE Ingredient
GO	