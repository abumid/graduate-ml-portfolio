/*DRPT Coursework by 001516, 001631, 001805*/
use KFCDB
go
----Task 2.4. Export data from database
----Exporting to XML
--version 1
select 
	p.Name as ProductName, 
	p.Price as ProductPrice,
	p.ProductType as ProductType,
	p.Description as ProductInformation,
	fc.Name as Category,
	od.Quantity as OrderedQuantity,
	o.OrderDate as OrderDate,
	r.Name as OrderedRestaurant,
	r.City as RestaurantLocation
from Product p
	join FoodCategory fc on p.CategoryId = fc.CategoryId
	join OrderDetail od on p.ProductId = od.ProductId
	join [Order] o on o.OrderId = od.OrderId
	join Restaurant r on r.RestaurantId = o.RestaurantId
	where 
		(p.Price > 7000 or
		fc.name = 'Burger' or 
		(od.Quantity between 2 and 7)) and
		p.Vegetarian = 1
for xml auto, root('KFC_report')
go

--version 2
create or alter function udfProductSalestoXML() 
returns xml 
as
begin
	declare @XML xml = (
		select 
			p.Name as ProductName, 
			p.Price as ProductPrice,
			p.ProductType as ProductType,
			p.Description as ProductInformation,
			fc.Name as Category,
			od.Quantity as OrderedQuantity,
			o.OrderDate as OrderDate,
			r.Name as OrderedRestaurant,
			r.City as RestaurantLocation
	from Product p
	join FoodCategory fc on p.CategoryId = fc.CategoryId
	join OrderDetail od on p.ProductId = od.ProductId
	join [Order] o on o.OrderId = od.OrderId
	join Restaurant r on r.RestaurantId = o.RestaurantId
	where 
		(p.Price > 7000 or
		fc.name = 'Burger' or 
		(od.Quantity between 2 and 7)) and
		p.Vegetarian = 1
	for xml path('ProductSales'), ROOT('Product'), type
		)
return @XML
end
go
--test XML export
select dbo.udfProductSalestoXML()
go

----Exporting to JSON

select p.Name "Product name", 
	   p.Price "Product price (UZS)",
	   p.ProductType "Product type",
	   p.Description "Product information",
	   fc.Name "Category",
	   od.Quantity "Quantity of product in order",
	   o.OrderDate "Order date",
	   r.Name "Product ordered Restaurant",
	   r.City "Restaurant location"
	from Product p
	join FoodCategory fc on p.CategoryId = fc.CategoryId
	join OrderDetail od on p.ProductId = od.ProductId
	join [Order] o on o.OrderId = od.OrderId
	join Restaurant r on r.RestaurantId = o.RestaurantId
	where 
		(p.Price > 7000 or
		fc.name = 'Burger' or 
		(od.Quantity between 2 and 7)) and
		p.Vegetarian = 1
for json auto, root('KFC-report')
go

--version 2
create or alter function udfProductSalestoJson() 
returns varchar(max)
as
begin
  declare @json varchar(max) = (  
		select 
			p.Name as ProductName, 
			p.Price as ProductPrice,
			p.ProductType as ProductType,
			p.Description as ProductInformation,
			fc.Name as Category,
			od.Quantity as OrderedQuantity,
			o.OrderDate as OrderDate,
			r.Name as OrderedRestaurant,
			r.City as RestaurantLocation
		from Product p
			join FoodCategory fc on p.CategoryId = fc.CategoryId
			join OrderDetail od on p.ProductId = od.ProductId
			join [Order] o on o.OrderId = od.OrderId
			join Restaurant r on r.RestaurantId = o.RestaurantId
		where 
			(p.Price > 7000 or
			fc.name = 'Burger' or 
			(od.Quantity between 2 and 7)) and
			p.Vegetarian = 1
		for json path
	)
return @json
end
go

--test Json export
select dbo.udfProductSalestoJson()
go

----Exporting to CSV
go
create or alter function ProductSalesToCsv()
returns varchar(max) as
begin 
  declare @csv varchar(max) = (
  select 
    string_agg(concat(
     p.Name, ',', 
     p.Price,',', 
	 p.ProductType, ',', 
     p.Description,',', 
     fc.Name,',', 
     od.Quantity, ',',
	 o.OrderDate,',', 
	 r.Name,',',	 
	 r.City), char(13))
	from Product p
	join FoodCategory fc on p.CategoryId = fc.CategoryId
	join OrderDetail od on p.ProductId = od.ProductId
	join [Order] o on o.OrderId = od.OrderId
	join Restaurant r on r.RestaurantId = o.RestaurantId
	where (p.Price > 7000 or
	fc.name = 'Burger' or 
	(od.Quantity between 2 and 7)) and
	p.Vegetarian = 1
  )
  return @csv
end
go
select dbo.ProductSalesToCsv()