/*DRPT Coursework by 001516, 001631, 001805*/
USE master
GO
--DROP DATABASE KFCDB
--GO
CREATE DATABASE KFCDB
GO
USE KFCDB
GO

--1. Create Customer table
CREATE TABLE Customer (
	CustomerId int identity(1,1) NOT NULL,
	FirstName nvarchar(50) NOT NULL,
	LastName nvarchar(50) NOT NULL,
	Email nvarchar(50) NOT NULL,
	PhoneNumber nvarchar(18) NOT NULL,
	Street nvarchar(150),
	City nvarchar(50),
	Region nvarchar(50),
	Country nvarchar(50),
	ZipCode nvarchar(10),
	DateOfBirth datetime NOT NULL,
	RegisteredDate datetime NOT NULL,
 CONSTRAINT PK_Customer PRIMARY KEY (CustomerId),
 CONSTRAINT CHK_Customer_Zip Check (ZipCode like '______'),
 CONSTRAINT CHK_Customer_PhoneNumber Check (PhoneNumber like '+(___)__-___-__-__')
 )
 GO

-- 2. Create Position table
 CREATE TABLE Position(
	PoistionId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	Description nvarchar(500) NULL,
 CONSTRAINT PK_Position PRIMARY KEY (PoistionId),
 CONSTRAINT UCHK_Position_Name Unique(Name)
) 

-- 3. Create Restaurant table
CREATE TABLE Restaurant(
	RestaurantId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	PhoneNumber nvarchar(50),
	Street nvarchar(150),
	City nvarchar(50),
	Region nvarchar(50),
	Country nvarchar(50),
	ZipCode nvarchar(10),
	ManagerId int,
 CONSTRAINT PK_Restaurant PRIMARY KEY (RestaurantId),
 CONSTRAINT CHK_Restaurant_Zip Check (ZipCode like '______')
)
GO

-- 4. Create Employee table
CREATE TABLE Employee(
	EmployeeId int identity(1,1) NOT NULL,
	FirstName nvarchar(50) NOT NULL,
	LastName nvarchar(50) NOT NULL,
	DateOfBirth datetime,
	HireDate datetime,
	Salary decimal(10, 2),
	PhoneNumber nvarchar(18),
	Email nvarchar(50),
	RestaurantId int NULL,
	PositionId int NOT NULL,
 CONSTRAINT PK_Employee PRIMARY KEY (EmployeeId),
 CONSTRAINT CHK_Employee_PhoneNumber Check (PhoneNumber like '+(___)__-___-__-__'),
 CONSTRAINT FK_Employee_Position FOREIGN KEY(PositionId) REFERENCES Position (PoistionId),
 CONSTRAINT FK_Employee_Restaurant FOREIGN KEY(RestaurantId) REFERENCES Restaurant (RestaurantId) 
)
GO

-- 5. Create EmployeeSchedule table
CREATE TABLE EmployeeSchedule(
	ScheduleId int identity(1,1) NOT NULL,
	ShiftStartTime datetime NOT NULL,
	ShiftEndTime datetime NOT NULL,
	DayOfWeek int NULL,
	ShiftDate date NULL,
	EmployeeId int NOT NULL,
 CONSTRAINT PK_EmployeeSchedule PRIMARY KEY CLUSTERED (ScheduleId),
 CONSTRAINT FK_EmployeeSchedule_Employee FOREIGN KEY(EmployeeId) REFERENCES Employee (EmployeeId)
)
GO

-- 6. Create Feedback table
CREATE TABLE Feedback(
	FeedbackId int identity(1,1) NOT NULL,
	RestaurantId int NOT NULL,
	CustomerId int NOT NULL,
	Rating int NULL,
	Comments nvarchar(500) NULL,
	SubmissionDate datetime NULL,
 CONSTRAINT PK_Feedback PRIMARY KEY (FeedbackId),
 CONSTRAINT FK_Feedback_Customer FOREIGN KEY(CustomerId) REFERENCES Customer (CustomerId),
 CONSTRAINT FK_Feedback_Restaurant FOREIGN KEY(RestaurantId) REFERENCES Restaurant (RestaurantId)
)
GO

-- 7. Create Order table
CREATE TABLE [Order](
	OrderId int identity(1,1) NOT NULL,
	CustomerId int NOT NULL,
	RestaurantId int NOT NULL,
	OrderDate datetime NOT NULL,
	TotalAmount decimal(7, 2),
	PaymentStatus nvarchar(10),
	DeliveryStatus nvarchar(10),
 CONSTRAINT PK_Order PRIMARY KEY (OrderId),
 CONSTRAINT FK_Order_Customer FOREIGN KEY(CustomerId) REFERENCES Customer (CustomerId),
 CONSTRAINT FK_Order_Restaurant FOREIGN KEY(RestaurantId) REFERENCES Restaurant (RestaurantId)
)
GO

-- 8. Create PaymentMethod table
CREATE TABLE PaymentMethod(
	MethodId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	CardType nvarchar(50),
	CardNumber nvarchar(16),
	ExpirationDate datetime,
	BillingAddress nvarchar(200),
 CONSTRAINT PK_PaymentMethod PRIMARY KEY (MethodId),
 CONSTRAINT CHK_PaymentMethod_CardNumber Check (CardNumber like '________________')
)
GO

-- 9. Create Transaction table
CREATE TABLE [Transaction](
	TransactionId int identity(1,1) NOT NULL,
	OrderId int NOT NULL,
	PaymentMethodId int NOT NULL,
	TransactionDate datetime NOT NULL,
	Amount decimal(7, 2) NOT NULL,
	TransactionStatus nvarchar(10),
 CONSTRAINT PK_Transaction PRIMARY KEY (TransactionId ASC),
 CONSTRAINT FK_Transaction_Order FOREIGN KEY(OrderId) REFERENCES [Order] (OrderId),
 CONSTRAINT FK_Transaction_PaymentMethod FOREIGN KEY(PaymentMethodId) REFERENCES PaymentMethod (MethodId)
)
GO

-- 10. Create Supplier table
CREATE TABLE [Supplier](
	SupplierId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	Street nvarchar(50),
	City nvarchar(50),
	Region nvarchar(50),
	Country nvarchar(50),
	ZipCode nvarchar(10),
 CONSTRAINT PK_Supplier PRIMARY KEY (SupplierId),
 CONSTRAINT CHK_Supplier_Zip Check (ZipCode like '______')
)
GO

-- 11. Create Ingredient table
CREATE TABLE [Ingredient](
	IngredientId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	UnitPrice decimal(7, 2),
	QuantityInStock int,
	ExpirationDate datetime,
CONSTRAINT PK_Ingredient PRIMARY KEY (IngredientId),
)
GO

-- 12. Create Supply table
CREATE TABLE [Supply](
	SupplyId int identity(1,1) NOT NULL,
	RestaurantId int NOT NULL,
	SupplierId int NOT NULL,
	IngredientId int NOT NULL,
	SupplyDate datetime NOT NULL,
	Quantity int,
	ExpiryDate datetime,
	TotalPrice decimal(7, 2),
 CONSTRAINT PK_Supply PRIMARY KEY (SupplyId),
 CONSTRAINT FK_Supply_Restaurant FOREIGN KEY (RestaurantId) REFERENCES Restaurant (RestaurantId),
 CONSTRAINT FK_Supply_Supplier FOREIGN KEY (SupplierId) REFERENCES Supplier (SupplierId),
 CONSTRAINT FK_Supply_Ingredient FOREIGN KEY (IngredientId) REFERENCES Ingredient (IngredientId)
 )
 GO

-- 13. Create SupplierContactPerson table
 CREATE TABLE [SupplierContactPerson](
	ContactPersonId int identity(1,1) NOT NULL,
	SupplierId int NOT NULL,
	FirstName nvarchar(50),
	LastName nvarchar(50),
	Email nvarchar(50),
	PhoneNumber nvarchar(18),
CONSTRAINT PK_SupplierContactPerson PRIMARY KEY (ContactPersonId),
CONSTRAINT FK_SupplierContactPerson_Supplier FOREIGN KEY (SupplierId) REFERENCES Supplier (SupplierId),
CONSTRAINT CHK_SupplierContactPerson_PhoneNumber Check (PhoneNumber like '+(___)__-___-__-__'),
CONSTRAINT CHK_SupplierContactPerson_Email CHECK(Email like '%@%'),
CONSTRAINT UCHK_SupplierContactPerson_Email Unique(Email)
)
GO

-- 14. Create FoodCategory table
CREATE TABLE [FoodCategory](
	CategoryId int NOT NULL,
	Name nvarchar(100) NOT NULL,
	Description nvarchar(500) NULL,
 CONSTRAINT PK_FoodCategory PRIMARY KEY (CategoryId),
 CONSTRAINT UCHK_FoodCategory_Name Unique(Name)
)
GO

-- 15. Create Product table
CREATE TABLE [Product](
	ProductId int identity(1,1) NOT NULL,
	Name nvarchar(50) NOT NULL,
	Description nvarchar(300),
	Price decimal(7, 2) NOT NULL,
	ProductType nvarchar(10) NOT NULL,
	PreparationTimeInSeconds int,
	Vegetarian bit NOT NULL,
	CategoryId int NOT NULL,
CONSTRAINT PK_Product PRIMARY KEY (ProductId),
CONSTRAINT FK_Product_FoodCategory FOREIGN KEY (CategoryId) REFERENCES FoodCategory (CategoryId)
)
GO

-- 16. Create OrderDetail table
CREATE TABLE [OrderDetail](
	OrderId int NOT NULL,
	ProductId int NOT NULL,
	Quantity int NOT NULL,
	UnitPrice decimal(7, 2),
	TotalPriceForProduct decimal(7, 2),
CONSTRAINT FK_OrderDetail_Order FOREIGN KEY (OrderId) REFERENCES [Order] (OrderId),
CONSTRAINT FK_OrderDetail_Product FOREIGN KEY (ProductId) REFERENCES Product (ProductId),
)
GO

ALTER TABLE Restaurant ADD CONSTRAINT FK_Restaurant_Manager foreign key(ManagerId) references Employee(EmployeeID)
ON DELETE SET NULL
