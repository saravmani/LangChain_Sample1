# Functional Specification Document: Banking Portal API

## 1. Introduction

This document outlines the functional specifications for the Banking Portal API, as defined by the provided Swagger document. This API provides functionality for user authentication, account management, and transaction processing.

## 2. Purpose

The purpose of this API is to provide a secure and efficient interface for banking portal functionalities, enabling users to manage their accounts, perform transactions, and access relevant information.

## 3. Scope

This document covers the functionalities defined in the Swagger document, including user management (registration, login, update), account management (deposit, withdrawal, fund transfer, PIN management), and dashboard access.

## 4. Functional Requirements

### 4.1. User Management

* **4.1.1. User Registration (`/api/users/register` - POST)**
    * **Description:** Allows new users to register with the banking portal from India. With country code IN
    * **Business Validations:**
        * For the validation error s API should return 400 bad request error
        * The  Phone number  cannot be duplicated,  If Phone number exists then API should return  the Error Message. (Not JSON)  - Phone number already exists
        * The email id cannot be duplicated, If Email already exists then API should return the Error Message as simple string. (Not JSON) - Email already exists
        * Password should contain minimum 10 chars. Atleast 1 special character and 1 Capital letter and one Number and one small letter. If password policy not matches then API should return 400 response.
        
    * **Input:** A JSON object conforming to the `User` schema, containing:
        * `name` (string, required) - This can be any valid user name
        * `password` (string, required)
        * `email` (string, required) - This should be any valid email id Ex:my_mailid.5274@gmail.com
        * `countryCode` (string, required)
        * `phoneNumber` (string, required)
        * `address` (string, required)
    * **Output:** 
        * On successfull registration should return 200 status code and a json result with ifscCode, branch, accountType and all the input parameters except password
            Ex: {"name":"<given user name>","email":"<given email id>","countryCode":"IN","phoneNumber":"<given 10 digit mobile no>","address":"123 Paris Street, Paris","accountNumber":"<accountNumber>","ifscCode":"<ifscCode>","branch":"<branch>","accountType":"Savings"}


    * **Security:** Requires no authentication.
* **4.1.2. User Login (`/api/users/login` - POST)**
    * **Description:** Authenticates users and generates a session token.
    * **Input:** A JSON object conforming to the `LoginRequest` schema, containing:
        * `identifier` (string, required) - represents username or email.
        * `password` (string, required)
    * **Output:** On Successful login the API will return a json { "token": "<bearer token value>" }
    * **Security:** Requires no authentication.


* **4.1.3. User Update 
    API URL - `/api/users/update` - POST
    * **Description:** Allows users to update their profile information.
    * User ony can update Email. not the other details
    * **Input:** A JSON object conforming to the `User` schema, containing the updated user information.
        * Below fields are required for an API
            "name": "string",
            "password": "string",
            "email": "string",
            "countryCode": "string",
            "phoneNumber": "string",
            "address": "string",
    * **Output:** A string indicating the success or failure of the update.
    * **Security:** Requires Bearer token authentication.


* **4.1.4. Password Reset (`/api/auth/password-reset` - POST)**
    * **Description:** Allows users to reset their password.
    * **Input:** A JSON object conforming to the `ResetPasswordRequest` schema, containing:
        * `identifier` (string, required) - represents username or email.
        * `newPassword` (string, required)
    * **Output:** A string indicating the success or failure of the password reset.
    * **Security:** Requires no authentication.

### 4.2. Account Management

* **4.2.1. Cash Deposit (`/api/account/deposit` - POST)**    
    * Need to login to the applicaiton using api/users/login before performing this  
    * **Description:** Helps user to deposit the money
        * Validations  
            1. Amount cannot be greater than 100,000
            2. Amount must be in multiples of 100
            3. Amount must be greater than 0
            4. Invalid amount
            
    * **Input:** A JSON object conforming to the `AmountRequest` schema, containing:
        * `accountNumber` (string, required)
        * `pin` (string, required)
        * `amount` (number, double, required)
    * **Output:** 
        * On successful deplosit applicaiton will return 200 status code. 
        * If in valid authentication token porvided then application will return just 401 error message
    * **Security:** Requires Bearer token authentication. 
* **4.2.2. Cash Withdrawal (`/api/account/withdraw` - POST)**
    * Need to login to the applicaiton using api/users/login before performing this  
    * **Description:** Allows users to withdraw funds from their account.
    * **Input:** A JSON object conforming to the `AmountRequest` schema, containing:
        * `accountNumber` (string, required)
        * `pin` (string, required)
        * `amount` (number,  required)  - The amount should be in multiplies of 100
    * **Output:** A string indicating the success or failure of the withdrawal.
    * **Security:** Requires Bearer token authentication.
* **4.2.3. Fund Transfer (`/api/account/fund-transfer` - POST)**
    * **Description:** Allows users to transfer funds between accounts.
    * **Input:** A JSON object conforming to the `FundTransferRequest` schema, containing:
        * `sourceAccountNumber` (string, required)
        * `targetAccountNumber` (string, required)
        * `amount` (number, double, required) - The amount should be in multiplies of 100 s
        * `pin` (string, required)
    * **Output:** A string indicating the success or failure of the fund transfer.
    * **Security:** Requires Bearer token authentication.
* **4.2.4. Create PIN (`/api/account/pin/create` - POST)**
    * **Description:** Allows users to create a PIN for their account.
    * **Input:** A JSON object conforming to the `PinRequest` schema, containing:
        * `accountNumber` (string, required)
        * `pin` (string, required)
        * `password` (string, required)
    * **Output:** A string indicating the success or failure of PIN creation.
    * **Security:** Requires Bearer token authentication.
* **4.2.5. Update PIN (`/api/account/pin/update` - POST)**
    * **Description:** Allows users to update their account PIN.
    * **Input:** A JSON object conforming to the `PinUpdateRequest` schema, containing:
        * `accountNumber` (string, required)
        * `oldPin` (string, required)
        * `newPin` (string, required)
        * `password` (string, required)
    * **Output:** A string indicating the success or failure of the PIN update.
    * **Security:** Requires Bearer token authentication.
* **4.2.6. Check PIN (`/api/account/pin/check` - GET)**
    * **Description:** Allows users to check if their account PIN is valid.
    * **Output:** A string indicating the validity of the PIN.
    * **Security:** Requires Bearer token authentication.
* **4.2.7. Get All Transactions (`/api/account/transactions` - GET)**
    * **Description:** Retrieves all transactions associated with an account.
    * **Output:** A string containing the transaction data.
    * **Security:** Requires Bearer token authentication.

### 4.3. Dashboard Access

* **4.3.1. Get User Details (`/api/dashboard/user` - GET)**
    * **Description:** Retrieves user details for the dashboard.
    * **Output:** A string containing user details.
    * **Security:** Requires Bearer token authentication.
* **4.3.2. Get Account Details (`/api/dashboard/account` - GET)**
    * **Description:** Retrieves account details for the dashboard.
    * **Output:** A string containing account details.
    * **Security:** Requires Bearer token authentication.

## 5. Data Structures

* **User:** Represents user information.
* **Account:** Represents account information of the user **
* **Token:** Represents authentication token information.
* **LoginRequest:** Represents login credentials.
* **ResetPasswordRequest:** Represents password reset request.
* **AmountRequest:** Represents amount-related requests (deposit, withdrawal).
* **PinUpdateRequest:** Represents PIN update requests.
* **PinRequest:** Represents PIN creation request.
* **FundTransferRequest:** Represents fund transfer requests.

## 6. Security

* Bearer token authentication is required for all account management and dashboard access endpoints.
* Password hashing and secure storage must be implemented.
* Input validation and sanitization must be performed to prevent security vulnerabilities.
