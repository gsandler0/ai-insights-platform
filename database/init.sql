-- Initialize AI Insights Database

-- Create sales data table
CREATE TABLE sales_data (
    id SERIAL PRIMARY KEY,
    transaction_date DATE NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    sales_rep VARCHAR(100),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create customer data table
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    company VARCHAR(255),
    industry VARCHAR(100),
    region VARCHAR(50),
    registration_date DATE,
    customer_tier VARCHAR(20) DEFAULT 'Standard',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create product data table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),
    cost_price DECIMAL(10,2),
    list_price DECIMAL(10,2),
    margin_percent DECIMAL(5,2),
    supplier VARCHAR(255),
    stock_level INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample customer data
INSERT INTO customers (customer_id, customer_name, email, phone, company, industry, region, registration_date, customer_tier) VALUES
('CUST001', 'John Smith', 'john.smith@techcorp.com', '+1-555-0101', 'TechCorp Inc', 'Technology', 'North America', '2023-01-15', 'Premium'),
('CUST002', 'Sarah Johnson', 'sarah.j@innovate.com', '+1-555-0102', 'Innovate Solutions', 'Consulting', 'North America', '2023-02-20', 'Standard'),
('CUST003', 'Michael Chen', 'mchen@globaltech.com', '+1-555-0103', 'GlobalTech Ltd', 'Manufacturing', 'Asia Pacific', '2023-01-10', 'Premium'),
('CUST004', 'Emma Wilson', 'ewilson@startup.co', '+1-555-0104', 'Startup Co', 'Technology', 'Europe', '2023-03-05', 'Standard'),
('CUST005', 'David Brown', 'dbrown@enterprise.com', '+1-555-0105', 'Enterprise Corp', 'Finance', 'North America', '2023-01-25', 'Premium'),
('CUST006', 'Lisa Davis', 'ldavis@retail.com', '+1-555-0106', 'Retail Solutions', 'Retail', 'North America', '2023-02-15', 'Standard'),
('CUST007', 'Robert Garcia', 'rgarcia@logistics.com', '+1-555-0107', 'Logistics Pro', 'Transportation', 'South America', '2023-01-30', 'Standard'),
('CUST008', 'Anna Mueller', 'amueller@german.de', '+49-555-0108', 'German Industries', 'Manufacturing', 'Europe', '2023-02-10', 'Premium'),
('CUST009', 'James Taylor', 'jtaylor@consulting.com', '+1-555-0109', 'Taylor Consulting', 'Consulting', 'North America', '2023-03-01', 'Premium'),
('CUST010', 'Maria Rodriguez', 'mrodriguez@tech.mx', '+52-555-0110', 'Tech Mexico', 'Technology', 'South America', '2023-02-28', 'Standard');

-- Insert sample product data
INSERT INTO products (product_name, category, subcategory, cost_price, list_price, margin_percent, supplier, stock_level) VALUES
('Enterprise Software License', 'Software', 'Enterprise', 2000.00, 5000.00, 60.00, 'SoftwareVendor Inc', 100),
('Cloud Storage Plan', 'Services', 'Cloud', 50.00, 199.99, 75.00, 'CloudProvider Ltd', 999),
('Professional Services', 'Services', 'Consulting', 100.00, 250.00, 60.00, 'Internal', 50),
('Hardware Server', 'Hardware', 'Servers', 3000.00, 6000.00, 50.00, 'ServerTech Corp', 25),
('Security Suite', 'Software', 'Security', 500.00, 1200.00, 58.33, 'SecureTech Inc', 75),
('Training Program', 'Services', 'Training', 200.00, 800.00, 75.00, 'Training Corp', 30),
('Mobile App License', 'Software', 'Mobile', 300.00, 899.99, 66.67, 'AppDev Solutions', 200),
('Data Analytics Tool', 'Software', 'Analytics', 800.00, 2400.00, 66.67, 'Analytics Pro', 40),
('Support Contract', 'Services', 'Support', 150.00, 500.00, 70.00, 'Internal', 100),
('Custom Integration', 'Services', 'Development', 500.00, 2000.00, 75.00, 'Internal', 20);

-- Insert sample sales data
INSERT INTO sales_data (transaction_date, customer_id, product_name, category, quantity, unit_price, total_amount, sales_rep, region) VALUES
-- January 2024
('2024-01-15', 'CUST001', 'Enterprise Software License', 'Software', 2, 5000.00, 10000.00, 'Alice Cooper', 'North America'),
('2024-01-16', 'CUST002', 'Cloud Storage Plan', 'Services', 5, 199.99, 999.95, 'Bob Williams', 'North America'),
('2024-01-18', 'CUST003', 'Hardware Server', 'Hardware', 1, 6000.00, 6000.00, 'Charlie Kim', 'Asia Pacific'),
('2024-01-20', 'CUST004', 'Security Suite', 'Software', 3, 1200.00, 3600.00, 'Diana Prince', 'Europe'),
('2024-01-22', 'CUST005', 'Professional Services', 'Services', 10, 250.00, 2500.00, 'Alice Cooper', 'North America'),

-- February 2024
('2024-02-01', 'CUST006', 'Mobile App License', 'Software', 4, 899.99, 3599.96, 'Bob Williams', 'North America'),
('2024-02-03', 'CUST007', 'Training Program', 'Services', 2, 800.00, 1600.00, 'Eva Martinez', 'South America'),
('2024-02-05', 'CUST008', 'Data Analytics Tool', 'Software', 1, 2400.00, 2400.00, 'Diana Prince', 'Europe'),
('2024-02-10', 'CUST009', 'Support Contract', 'Services', 8, 500.00, 4000.00, 'Alice Cooper', 'North America'),
('2024-02-12', 'CUST010', 'Custom Integration', 'Services', 1, 2000.00, 2000.00, 'Eva Martinez', 'South America'),

-- March 2024
('2024-03-01', 'CUST001', 'Cloud Storage Plan', 'Services', 10, 199.99, 1999.90, 'Alice Cooper', 'North America'),
('2024-03-05', 'CUST003', 'Security Suite', 'Software', 2, 1200.00, 2400.00, 'Charlie Kim', 'Asia Pacific'),
('2024-03-08', 'CUST005', 'Enterprise Software License', 'Software', 1, 5000.00, 5000.00, 'Alice Cooper', 'North America'),
('2024-03-10', 'CUST002', 'Professional Services', 'Services', 15, 250.00, 3750.00, 'Bob Williams', 'North America'),
('2024-03-15', 'CUST008', 'Hardware Server', 'Hardware', 2, 6000.00, 12000.00, 'Diana Prince', 'Europe'),

-- April 2024
('2024-04-02', 'CUST004', 'Training Program', 'Services', 3, 800.00, 2400.00, 'Diana Prince', 'Europe'),
('2024-04-05', 'CUST006', 'Data Analytics Tool', 'Software', 2, 2400.00, 4800.00, 'Bob Williams', 'North America'),
('2024-04-08', 'CUST009', 'Mobile App License', 'Software', 6, 899.99, 5399.94, 'Alice Cooper', 'North America'),
('2024-04-12', 'CUST007', 'Support Contract', 'Services', 5, 500.00, 2500.00, 'Eva Martinez', 'South America'),
('2024-04-15', 'CUST010', 'Enterprise Software License', 'Software', 1, 5000.00, 5000.00, 'Eva Martinez', 'South America'),

-- May 2024
('2024-05-01', 'CUST003', 'Custom Integration', 'Services', 2, 2000.00, 4000.00, 'Charlie Kim', 'Asia Pacific'),
('2024-05-05', 'CUST001', 'Professional Services', 'Services', 20, 250.00, 5000.00, 'Alice Cooper', 'North America'),
('2024-05-08', 'CUST008', 'Cloud Storage Plan', 'Services', 15, 199.99, 2999.85, 'Diana Prince', 'Europe'),
('2024-05-10', 'CUST005', 'Security Suite', 'Software', 4, 1200.00, 4800.00, 'Alice Cooper', 'North America'),
('2024-05-15', 'CUST002', 'Hardware Server', 'Hardware', 1, 6000.00, 6000.00, 'Bob Williams', 'North America'),

-- June 2024
('2024-06-02', 'CUST006', 'Training Program', 'Services', 4, 800.00, 3200.00, 'Bob Williams', 'North America'),
('2024-06-05', 'CUST009', 'Data Analytics Tool', 'Software', 1, 2400.00, 2400.00, 'Alice Cooper', 'North America'),
('2024-06-08', 'CUST004', 'Mobile App License', 'Software', 8, 899.99, 7199.92, 'Diana Prince', 'Europe'),
('2024-06-12', 'CUST007', 'Enterprise Software License', 'Software', 1, 5000.00, 5000.00, 'Eva Martinez', 'South America'),
('2024-06-15', 'CUST010', 'Support Contract', 'Services', 10, 500.00, 5000.00, 'Eva Martinez', 'South America');

-- Create indexes for better performance
CREATE INDEX idx_sales_data_date ON sales_data(transaction_date);
CREATE INDEX idx_sales_data_customer ON sales_data(customer_id);
CREATE INDEX idx_sales_data_category ON sales_data(category);
CREATE INDEX idx_customers_region ON customers(region);
CREATE INDEX idx_customers_tier ON customers(customer_tier);
CREATE INDEX idx_products_category ON products(category);

-- Create a view for sales analytics
CREATE VIEW sales_summary AS
SELECT 
    s.transaction_date,
    s.customer_id,
    c.customer_name,
    c.company,
    c.industry,
    c.region,
    c.customer_tier,
    s.product_name,
    s.category,
    s.quantity,
    s.unit_price,
    s.total_amount,
    s.sales_rep,
    p.cost_price,
    p.margin_percent,
    (s.total_amount * p.margin_percent / 100) as profit_amount
FROM sales_data s
JOIN customers c ON s.customer_id = c.customer_id
LEFT JOIN products p ON s.product_name = p.product_name;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO insights_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO insights_user;
GRANT SELECT ON sales_summary TO insights_user;

-- Create function to get table schema information
CREATE OR REPLACE FUNCTION get_table_schema(table_name_param text)
RETURNS TABLE(
    column_name text,
    data_type text,
    is_nullable text,
    column_default text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.column_name::text,
        c.data_type::text,
        c.is_nullable::text,
        c.column_default::text
    FROM information_schema.columns c
    WHERE c.table_name = table_name_param
    AND c.table_schema = 'public'
    ORDER BY c.ordinal_position;
END;
$$ LANGUAGE plpgsql;
