/*DRPT Coursework by 001516, 001631, 001805*/
USE KFCDB
go
-----------------------------------------------------------------
--Import data CSV
select string_agg(CONCAT((d.ProductId), ',', d.Name, ',', d.Description, ',', d.Price, ',', d.ProductType, ',', d.PreparationTimeInSeconds, ',', d.Vegetarian, ',', d.CategoryId), char(13)) 
from Product d
go 
--version 1
create or alter procedure udpProductsCsvImport(@csvStr nvarchar(max))
as
begin 
	with e as (
		select trim(value) csvLine
		from string_split(@csvStr, char(13))
		where trim(value) <> ''
	),
	splitted as (
		select *
		, lead(value, 0) over (partition by csvLine order by csvLine) as name,		
		lead(value, 1) over (partition by csvLine order by csvLine) as description,
		lead(value, 2) over (partition by csvLine order by csvLine) as price,		
		lead(value, 3) over (partition by csvLine order by csvLine) as product_type,
		lead(value, 4) over (partition by csvLine order by csvLine) as prep_time,
		lead(value, 5) over (partition by csvLine order by csvLine) as isVegetarian,
		lead(value, 6) over (partition by csvLine order by csvLine) as category_id,
		row_number() over(partition by csvLine order by csvLine) as idx 
		from e cross apply string_split(csvLine, ',')
	)
	insert into Product(Name, Description, Price, ProductType,PreparationTimeInSeconds,Vegetarian,CategoryId)
	select name, description, price, product_type, prep_time, isVegetarian, category_id
	from splitted
	where idx = 1
end
go
exec udpProductsCsvImport
	@csvStr = 'Twister,TwisterDescription,18500.00,Food,1200,0,9
Twister Eco,TwisterEcoDescription,15750.00,Food,450,1,9
Coca cola,Coca-cola drink description,10000.00,Beverage,500,1,5
Sanders Basket,Sanders basket description,45000.00,Food,500,0,1'
go
select * from Product
go

--version 2
BULK INSERT Product 
FROM 'D:\Data.csv' WITH ( FORMAT='CSV', FIRSTROW=1, FIELDTERMINATOR = ',', ROWTERMINATOR = '\n', KEEPIDENTITY, MAXERRORS = 0)

/*Content of data.csv  file:*/
/*
51,Twister,TwisterDescription,18500.00,Food,1200,0,9
52,Twister Eco,TwisterEcoDescription,15750.00,Food,450,1,9
53,Coca cola,Coca-cola drink description,10000.00,Beverage,500,1,5
54,Sanders Basket,Sanders basket description,45000.00,Food,500,0,1
*/

--check result
select * from Product

--------------------------------------------------------------------
-- Import from XML text
DECLARE @xmlData XML;

SET @xmlData = 
'<Products>
    <Product>
        <ProductId>201</ProductId>
        <Name>Sanders burger</Name>
        <Description>Description1</Description>
        <Price>35000.50</Price>
        <ProductType>Food</ProductType>
        <PreparationTimeInSeconds>300</PreparationTimeInSeconds>
        <Vegetarian>1</Vegetarian>
        <CategoryId>3</CategoryId>
    </Product>
    <Product>
        <ProductId>202</ProductId>
        <Name>Sandwich</Name>
        <Description>Description2</Description>
        <Price>15000.75</Price>
        <ProductType>Food</ProductType>
        <PreparationTimeInSeconds>450</PreparationTimeInSeconds>
        <Vegetarian>0</Vegetarian>
        <CategoryId>4</CategoryId>
    </Product>
</Products>';

DECLARE @hDoc INT;
EXEC sp_xml_preparedocument @hDoc OUTPUT, @xmlData;

INSERT INTO Product (Name, Description, Price, ProductType, PreparationTimeInSeconds, Vegetarian, CategoryId)
SELECT
    Name,
    Description,
    Price,
    ProductType,
    PreparationTimeInSeconds,
    Vegetarian,
    CategoryId
FROM OPENXML(@hDoc, '/Products/Product', 2)
WITH (
    ProductId int 'ProductId',
    Name nvarchar(50) 'Name',
    Description nvarchar(300) 'Description',
    Price decimal(7, 2) 'Price',
    ProductType nvarchar(10) 'ProductType',
    PreparationTimeInSeconds int 'PreparationTimeInSeconds',
    Vegetarian bit 'Vegetarian',
    CategoryId int 'CategoryId'
);

EXEC sp_xml_removedocument @hDoc;
go

select * from Product
go
-------------------------------------------------------------------------
-- Import from JSON text
create or alter procedure udpProductJsonImport(@jsonStr nvarchar(max))
as 
begin 
	insert into Product (Name, Description, Price, ProductType, PreparationTimeInSeconds, Vegetarian, CategoryId)
	select *
	from OPENJSON(@jsonStr, '$.Products')
		with(
			Name nvarchar(50) '$.name',
			Description nvarchar(300) '$.description',
			Price decimal(7, 2) '$.price',
			ProductType nvarchar(10) '$.product_type',
			PreparationTimeInSeconds int '$.preparation_time',
			Vegetarian bit '$.vegetarian',
			CategoryId int '$.category_id'
		)		
end
go
exec udpProductJsonImport
	@jsonStr = 
		'{
		"Products": [
			{
				"name": "Green tea",
				"price": "7500.00",
				"description": "Imported from Turkey",
				"product_type": "Beverage",
				"preparation_time": "1200",
				"vegetarian": "1",
				"category_id": "5"
			},
			{
				"name": "Fanta",
				"price": "12000.00",
				"description": "Cold drink",
				"product_type": "Beverage",
				"preparation_time": "50",
				"vegetarian": "1",
				"category_id": "5"
			},
			{
				"name": "Spicy wings 5",
				"price": "50000.00",
				"description": "Chicken wings",
				"product_type": "Food",
				"preparation_time": "1200",
				"vegetarian": "0",
				"category_id": "1"
			},
			{
				"name": "Basket No1",
				"price": "42000.00",
				"description": "Chicken basket: wings, file",
				"product_type": "Food",
				"preparation_time": "2400",
				"vegetarian": "0",
				"category_id": "8"
			},
			{
				"name": "Green burger",
				"price": "27500.00",
				"description": "Vegan burger",
				"product_type": "Food",
				"preparation_time": "1200",
				"vegetarian": "1",
				"category_id": "2"
			}
		]
	}'

-- check the results
select * from Product
