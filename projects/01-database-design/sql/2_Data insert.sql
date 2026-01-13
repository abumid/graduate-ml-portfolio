
use KFCDB
go

INSERT INTO Customer (FirstName, LastName, Email, PhoneNumber, Street, City, Region, Country, ZipCode, DateOfBirth, RegisteredDate)
VALUES
    ('Jamoliddin', 'Olimov', 'jamoliddin@gmail.com', '+(998)90-999-99-99', '123 Navoi Street', 'Tashkent', 'Tashkent', 'Uzbekistan', '100100', '1990-07-12', '2023-10-01'),
    ('Dilrabo', 'Saidova', 'dilrabo@gmail.com', '+(998)95-678-12-34', '456 Babur Avenue', 'Samarkand', 'Samarkand', 'Uzbekistan', '200200', '1988-02-18', '2023-10-05'),
    ('Shokhida', 'Xolmatova', 'shokhida@gmail.com', '+(998)97-111-22-33', '789 Mustakillik St', 'Bukhara', 'Bukhara', 'Uzbekistan', '300300', '1995-09-30', '2023-10-10'),
    ('Islom', 'Toshmatov', 'islom@gmail.com', '+(998)91-987-65-43', '321 Amir Temur St', 'Andijan', 'Andijan', 'Uzbekistan', '400400', '1987-04-04', '2023-10-15'),
    ('Nargiza', 'Qosimova', 'nargiza@gmail.com', '+(998)94-333-99-99', '567 Babur Ave', 'Fergana', 'Fergana', 'Uzbekistan', '500500', '1993-12-28', '2023-10-20'),
    ('Farrukh', 'Rahmonov', 'farrukh@gmail.com', '+(998)93-555-77-77', '789 Navoi Street', 'Tashkent', 'Tashkent', 'Uzbekistan', '600600', '1986-06-15', '2023-10-25'),
    ('Dilobar', 'Usmonova', 'dilobar@gmail.com', '+(998)96-444-88-88', '456 Jizzakh Ave', 'Jizzakh', 'Jizzakh', 'Uzbekistan', '700700', '1994-03-21', '2023-11-01'),
    ('Murod', 'Xasanov', 'murod@gmail.com', '+(998)92-666-11-11', '321 Termez Street', 'Termez', 'Surkhandarya', 'Uzbekistan', '800800', '1989-09-10', '2023-11-05'),
    ('Zuhra', 'Orifova', 'zuhra@gmail.com', '+(998)99-777-00-00', '567 Gulistan Ave', 'Gulistan', 'Sirdaryo', 'Uzbekistan', '900900', '1992-12-01', '2023-11-10'),
	('Sirojiddin', 'Otabekov', 'sirojiddin@gmail.com', '+(998)98-888-22-22', '123 Khiva Street', 'Khiva', 'Khorezm', 'Uzbekistan', '101010', '1985-05-28', '2023-11-15');

INSERT INTO Position (Name, Description)
VALUES
    ('Manager', 'Responsible for overall restaurant management'),
    ('Assistant Manager', 'Assists the manager in daily operations'),
    ('Cashier', 'Handles customer transactions at the counter'),
    ('Cook', 'Prepares and cooks food items'),
    ('Server', 'Serves food and assists customers in the dining area'),
    ('Cleaner', 'Responsible for maintaining cleanliness in the restaurant'),
    ('Security Guard', 'Ensures the safety and security of the premises'),
    ('Shift Supervisor', 'Supervises the staff during specific shifts'),
    ('Delivery Driver', 'Delivers food orders to customers'),
    ('Kitchen Helper', 'Assists the cook in food preparation');

INSERT INTO Restaurant (Name, PhoneNumber, Street, City, Region, Country, ZipCode, ManagerId)
VALUES
    ('KFC Tashkent', '+(998)90-123-45-67', '123 Navoi Street', 'Tashkent', 'Tashkent', 'Uzbekistan', '100100', NULL),
    ('KFC Samarkand', '+(998)95-678-12-34', '456 Babur Avenue', 'Samarkand', 'Samarkand', 'Uzbekistan', '200200', NULL),
    ('KFC Bukhara', '+(998)97-111-22-33', '789 Mustakillik St', 'Bukhara', 'Bukhara', 'Uzbekistan', '300300', NULL),
    ('KFC Andijan', '+(998)91-987-65-43', '321 Amir Temur St', 'Andijan', 'Andijan', 'Uzbekistan', '400400', NULL),
    ('KFC Fergana', '+(998)94-333-99-99', '567 Babur Ave', 'Fergana', 'Fergana', 'Uzbekistan', '500500', NULL),
	('KFC Tashkent Oasis', '+(998)99-888-77-77', '789 Oasis Blvd', 'Tashkent', 'Tashkent', 'Uzbekistan', '938765', NULL),
    ('KFC Jizzakh', '+(998)96-444-88-88', '456 Jizzakh Ave', 'Jizzakh', 'Jizzakh', 'Uzbekistan', '700700', NULL),
    ('KFC Termez', '+(998)92-666-11-11', '321 Termez Street', 'Termez', 'Surkhandarya', 'Uzbekistan', '800800', NULL),
    ('KFC Gulistan', '+(998)99-777-00-00', '567 Gulistan Ave', 'Gulistan', 'Sirdaryo', 'Uzbekistan', '900900', NULL),
    ('KFC Khiva', '+(998)98-888-22-22', '123 Khiva Street', 'Khiva', 'Khorezm', 'Uzbekistan', '101010', NULL);

INSERT INTO Employee (FirstName, LastName, DateOfBirth, HireDate, Salary, PhoneNumber, Email, RestaurantId, PositionId)
VALUES
	('Jamoliddin', 'Olimov', '1990-01-15', '2022-01-01', 50000.00, '+(998)90-123-45-67', 'jamoliddin@email.com', 1, 1),
    ('Dilrabo', 'Saidova', '1992-05-20', '2022-02-01', 45000.00, '+(998)95-678-12-34', 'dilrabo@email.com', 2, 2),
	('Shokhida', 'Xolmatova', '1985-08-10', '2022-03-01', 55000.00, '+(998)97-111-22-33', 'shokhida@email.com', 3, 3),
	('Islom', 'Toshmatov', '1998-03-25', '2022-04-01', 48000.00, '+(998)91-987-65-43', 'islom@email.com', 4, 4),
	('Nargiza', 'Qosimova', '1982-11-08', '2022-05-01', 52000.00, '+(998)94-333-99-99', 'nargiza@email.com', 5, 5),
	('Farrukh', 'Rahmonov', '1990-07-12', '2022-06-01', 60000.00, '+(998)93-555-77-77', 'farrukh@email.com', 6, 6),
	('Dilobar', 'Usmonova', '1988-02-18', '2022-07-01', 58000.00, '+(998)96-444-88-88', 'dilobar@email.com', 7, 7),
    ('Murod', 'Xasanov', '1995-09-30', '2022-08-01', 53000.00, '+(998)92-666-11-11', 'murod@email.com', 8, 8),
    ('Zuhra', 'Orifova', '1987-04-04', '2022-09-01', 56000.00, '+(998)99-777-00-00', 'zuhra@email.com', 9, 9),
	('Sirojiddin', 'Otabekov', '1993-12-28', '2022-10-01', 49000.00, '+(998)98-888-22-22', 'sirojiddin@email.com', 10, 10);


INSERT INTO EmployeeSchedule (ShiftStartTime, ShiftEndTime, DayOfWeek, ShiftDate, EmployeeId)
VALUES
    ('2023-01-01 09:00:00', '2023-01-01 17:00:00', 7, '2023-01-01', 1),
    ('2023-01-02 10:00:00', '2023-01-02 18:00:00', 1, '2023-01-02', 2),
    ('2023-01-03 11:00:00', '2023-01-03 19:00:00', 2, '2023-01-03', 3),
    ('2023-01-04 12:00:00', '2023-01-04 20:00:00', 3, '2023-01-04', 4),
    ('2023-01-05 13:00:00', '2023-01-05 21:00:00', 4, '2023-01-05', 5),
    ('2023-01-06 14:00:00', '2023-01-06 22:00:00', 5, '2023-01-06', 6),
    ('2023-01-07 15:00:00', '2023-01-07 23:00:00', 6, '2023-01-07', 7),
    ('2023-01-08 09:00:00', '2023-01-08 17:00:00', 7, '2023-01-08', 8),
    ('2023-01-09 10:00:00', '2023-01-09 18:00:00', 1, '2023-01-09', 9),
    ('2023-01-10 11:00:00', '2023-01-10 19:00:00', 2, '2023-01-10', 10);

INSERT INTO Feedback (RestaurantId, CustomerId, Rating, Comments, SubmissionDate)
VALUES
    (1, 1, 5, 'Excellent service and delicious food!', '2023-01-15 12:30:00'),
    (2, 2, 4, 'The ambiance is great, but the wait time was a bit long.', '2023-01-16 18:45:00'),
    (3, 3, 3, 'Average experience, could improve cleanliness.', '2023-01-17 14:20:00'),
    (4, 4, 5, 'Outstanding! Will definitely visit again.', '2023-01-18 20:10:00'),
    (5, 5, 2, 'Disappointed with the food quality and service.', '2023-01-19 15:55:00'),
    (6, 6, 4, 'Friendly staff and good variety on the menu.', '2023-01-20 11:40:00'),
    (7, 7, 5, 'Exceptional experience, loved the special dish!', '2023-01-21 09:25:00'),
    (8, 8, 3, 'Waited too long for the order, but the food was decent.', '2023-01-22 16:30:00'),
    (9, 9, 4, 'Great atmosphere, enjoyed the live music!', '2023-01-23 13:15:00'),
    (10, 10, 5, 'Perfect in every aspect. Best dining experience!', '2023-01-24 19:05:00');

INSERT INTO [Order] (CustomerId, RestaurantId, OrderDate, TotalAmount, PaymentStatus, DeliveryStatus)
VALUES
    (1, 1, '2023-01-01 12:30:00', 25000.00, 'Paid', 'Delivered'),
    (2, 2, '2023-01-02 18:45:00', 40000.00, 'Paid', 'In Progres'),
    (3, 3, '2023-01-03 14:20:00', 30000.00, 'Pending', NULL),
    (4, 4, '2023-01-04 20:10:00', 50000.00, 'Paid', 'Delivered'),
    (5, 5, '2023-01-05 15:55:00', 15000.00, 'Paid', 'Delivered'),
    (6, 6, '2023-01-06 11:40:00', 35000.00, 'Paid', 'In Progres'),
    (7, 7, '2023-01-07 09:25:00', 60000.00, 'Paid', 'Delivered'),
    (8, 8, '2023-01-08 16:30:00', 20000.00, 'Pending', NULL),
    (9, 9, '2023-01-09 13:15:00', 45000.00, 'Paid', 'Delivered'),
    (10, 10, '2023-01-10 19:05:00', 75000.00, 'Paid', 'Delivered');


INSERT INTO PaymentMethod (Name, CardType, CardNumber, ExpirationDate, BillingAddress)
VALUES
    ('Jamoliddin Olimov', 'Uzcard', '1234567890123456', '2024-05-01', '123 Main Street, Tashkent, Uzbekistan'),
    ('Dilrabo Saidova', 'Humo', '2345678901234567', '2023-08-01', '456 Oak Avenue, Samarkand, Uzbekistan'),
    ('Shokhida Xolmatova', 'Visa', '3456789012345678', '2025-02-01', '789 Pine Road, Bukhara, Uzbekistan'),
    ('Islom Toshmatov', 'MasterCard', '4567890123456789', '2024-11-01', '101 Cedar Lane, Fergana, Uzbekistan'),
    ('Nargiza Karimova', 'Uzcard', '5678901234567890', '2023-04-01', '202 Maple Avenue, Namangan, Uzbekistan'),
    ('Akbar Nuriddinov', 'Humo', '6789012345678901', '2025-07-01', '303 Elm Street, Andijan, Uzbekistan'),
    ('Zulfiya Ismailova', 'Visa', '7890123456789012', '2023-10-01', '404 Birch Road, Nukus, Uzbekistan'),
    ('Farrukh Abdullayev', 'MasterCard', '8901234567890123', '2024-03-01', '505 Cedar Lane, Urgench, Uzbekistan'),
    ('Nodira Karimova', 'Uzcard', '9012345678901234', '2025-06-01', '606 Oak Avenue, Termez, Uzbekistan'),
    ('Sardor Tursunov', 'Humo', '0123456789012345', '2023-09-01', '707 Pine Road, Karshi, Uzbekistan');


INSERT INTO [Transaction] (OrderId, PaymentMethodId, TransactionDate, Amount, TransactionStatus)
VALUES
    (1, 1, '2023-01-01 13:00:00', 25000.00, 'Success'),
    (2, 2, '2023-01-02 19:00:00', 40000.00, 'Success'),
    (3, 3, '2023-01-03 15:30:00', 30000.00, 'Pending'),
    (4, 4, '2023-01-04 21:15:00', 50000.00, 'Success'),
    (5, 5, '2023-01-05 16:45:00', 15000.00, 'Success'),
    (6, 6, '2023-01-06 12:20:00', 35000.00, 'Pending'),
    (7, 7, '2023-01-07 10:10:00', 60000.00, 'Success'),
    (8, 8, '2023-01-08 17:00:00', 20000.00, 'Pending'),
    (9, 9, '2023-01-09 14:40:00', 45000.00, 'Success'),
    (10, 10, '2023-01-10 20:30:00', 75000.00, 'Success');


INSERT INTO [Supplier] (Name, Street, City, Region, Country, ZipCode)
VALUES
    ('ABC Elektronika', 'Abdulla Turdiev Street, 123', 'Tashkent', 'Tashkent', 'Uzbekistan', '100001'),
    ('Green Foods', 'Darvoza Street, 456', 'Samarkand', 'Samarkand', 'Uzbekistan', '200002'),
    ('Tech Solutions', 'Amir Temur Street, 789', 'Bukhara', 'Bukhara', 'Uzbekistan', '300003'),
    ('Fashion Trends', 'Sayram Street, 101', 'Fergana', 'Fergana', 'Uzbekistan', '400004'),
    ('Home Essentials', 'Namangan Street, 202', 'Namangan', 'Namangan', 'Uzbekistan', '500004'),
    ('Natures Best', 'Engels Street, 303', 'Andijan', 'Andijan', 'Uzbekistan', '600006'),
    ('Office Depot', 'Shota Rustaveli Street, 404', 'Nukus', 'Karakalpakstan', 'Uzbekistan', '700007'),
    ('Furniture Mart', 'K. Shirin Street, 505', 'Urgench', 'Khorezm', 'Uzbekistan', '800008'),
    ('Beauty Emporium', 'Dalvarza Street, 606', 'Termez', 'Surkhandarya', 'Uzbekistan', '900009'),
    ('Sporting Goods', 'Al-Buxoriy Street, 707', 'Qarshi', 'Kashkadarya', 'Uzbekistan', '910002');


INSERT INTO [Ingredient] (Name, UnitPrice, QuantityInStock, ExpirationDate)
VALUES
    ('Chicken Breast', 50000, 100, '2023-12-31'),
    ('Lettuce', 20000, 50, '2023-12-15'),
    ('Tomato', 10000, 75, '2023-12-20'),
    ('Cheese', 40000, 80, '2024-01-10'),
    ('Mayonnaise', 15000, 120, '2023-11-30'),
    ('Bread', 20000, 60, '2023-12-25'),
    ('Potatoes', 10000, 90, '2024-02-28'),
    ('Onions', 13000, 70, '2023-12-31'),
    ('Ketchup', 18000, 110, '2023-11-15'),
    ('Mustard', 16000, 100, '2023-11-20');

INSERT INTO [Supply] (RestaurantId, SupplierId, IngredientId, SupplyDate, Quantity, ExpiryDate, TotalPrice)
VALUES
    (1, 1, 1, '2023-01-15', 50, '2023-12-31', 25000.00),
    (2, 2, 2, '2023-02-20', 30, '2023-11-30', 60000.00),
    (3, 3, 3, '2023-03-25', 40, '2024-02-28', 40000.00),
    (4, 4, 4, '2023-04-10', 20, '2023-12-15', 80000.00),
    (5, 5, 5, '2023-05-18', 25, '2023-12-20', 50000.00),
    (1, 6, 6, '2023-06-12', 35, '2024-01-10', 35000.00),
    (2, 7, 7, '2023-07-05', 60, '2023-11-15', 72000.00),
    (3, 8, 8, '2023-08-30', 45, '2023-12-25', 54000.00),
    (4, 9, 9, '2023-09-08', 55, '2024-01-20', 44000.00),
    (5, 10, 10, '2023-10-14', 70, '2023-12-31', 70000.00);


INSERT INTO [SupplierContactPerson] (SupplierId, FirstName, LastName, Email, PhoneNumber)
VALUES
    (1, 'Javohir', 'Toshpulatov', 'javohir.toshpulatov@gmail.com', '+(998)99-123-45-67'),
    (2, 'Nigora', 'Xolmirzayeva', 'nigora.xolmirzayeva@gmail.com', '+(998)91-234-56-78'),
	(3, 'Muxlis', 'Qosimov', 'muxlis.qosimov@gmail.com', '+(998)93-345-67-89'),
	(4, 'Emiliya', 'Aminova', 'emiliya.aminova@gmail.com', '+(998)97-456-78-90'),
	(5, 'Daniyal', 'Xolmatov', 'daniyal.xolmatov@gmail.com', '+(998)95-567-89-01'),
	(6, 'Oliviya', 'Tursunova', 'oliviya.tursunova@gmail.com', '+(998)94-678-90-12'),
	(7, 'Uilyam', 'Sultonov', 'uilyam.sultonov@gmail.com', '+(998)98-789-01-23'),
	(8, 'Sofiya', 'Jamolova', 'sofiya.jamolova@gmail.com', '+(998)96-890-12-34'),
	(9, 'Jeyms', 'Shomurodov', 'jeyms.shomurodov@gmail.com', '+(998)92-901-23-45'),
	(10, 'Emma', 'Ergasheva', 'emma.ergasheva@gmail.com', '+(998)90-012-34-56');

INSERT INTO [FoodCategory] (CategoryId, Name, Description)
VALUES
    (1, 'Chicken Buckets', 'Satisfy your cravings with our signature chicken buckets'),
    (2, 'Burgers', 'Delicious and flavorful chicken burgers for every taste'),
    (3, 'Sides', 'Perfect accompaniments to your main course'),
    (4, 'Desserts', 'Indulge in sweet treats to complete your KFC experience'),
    (5, 'Drinks', 'Quench your thirst with our refreshing drink options'),
    (6, 'Salads', 'Fresh and crisp salads for a lighter option'),
    (7, 'Wraps', 'Enjoy the convenience of our tasty chicken wraps'),
    (8, 'Snack Boxes', 'Quick bites for a satisfying snack'),
    (9, 'Breakfast', 'Start your day right with our breakfast options'),
    (10, 'Kids Meals', 'Special meals crafted for our younger fans');

INSERT INTO [Product] (Name, Description, Price, ProductType, PreparationTimeInSeconds, Vegetarian, CategoryId)
VALUES
    ('Classic Chicken Burger', 'Classic chicken burger with lettuce and mayo', 50000.00, 'Food', 480, 0, 2),
    ('Spicy Chicken Wings', 'Spicy and crispy chicken wings', 45000.00, 'Food', 360, 0, 8),
    ('Vegetarian Wrap', 'Vegetarian wrap with assorted veggies', 49900.00, 'Food', 420, 1, 7),
    ('Family Bucket', 'Assorted chicken pieces for the whole family', 9000.00, 'Food', 900, 0, 1),
    ('Fries', 'Golden and crispy french fries', 24900.00, 'Food', 300, 1, 3),
    ('Cheese Pizza', 'Delicious cheese pizza with a crispy crust', 89900.00, 'Food', 600, 1, 9),
    ('Coleslaw', 'Crunchy coleslaw with a tangy dressing', 19900.00, 'Food', 240, 1, 6),
    ('Chocolate Sundae', 'Indulgent chocolate sundae for dessert', 32900.00, 'Food', 240, 0, 4),
    ('Iced Tea', 'Chilled iced tea for a refreshing drink', 17900.00, 'Beverage', 0, 1, 5),
    ('Breakfast Burrito', 'Hearty breakfast burrito with eggs and sausage', 47900.00, 'Food', 480, 0, 9);

INSERT INTO [OrderDetail] (OrderId, ProductId, Quantity, UnitPrice, TotalPriceForProduct)
VALUES
    (7, 1, 2, 5000.00, 10000.00),
    (7, 3, 1, 1990.00, 1990.00),
    (6, 2, 3, 4500.00, 13500.00),
    (6, 5, 1, 24900.00, 24900.00),
    (8, 4, 1, 5000.00, 89000.00),
    (8, 7, 2, 19900.00, 39800.00),
    (9, 6, 1, 89900.00, 89900.00),
    (10, 8, 2, 32900.00, 65800.00),
    (5, 10, 1, 47900.00, 47900.00),
    (5, 9, 1, 17900.00, 17900.00);

