
## Step 1: Create Project Folder

Open Command Prompt and navigate to your project directory:

```bash
cd D:\Docker_Practical\html-hello
```

---

## Step 2: Create Required Files

Create two files inside the folder:

- Dockerfile  
- index.html  

---

## Step 3: Build Docker Image

Run the following command to build the Docker image:

```bash
docker build -t html-hello .
```

---

## Step 4: Run Docker Container

Run container using:

```bash
docker run -d -p 8080:80 html-hello
```

---

## Step 5: Test in Browser

Open browser and go to:

http://localhost:8080

---

## Step 6: Check Docker Images

```bash
docker images
```

---

## Step 7: Check Running Containers

```bash
docker ps
```

---

## Step 8: Stop Running Container

```bash
docker stop 6d757a833842
```

