const canvas = document.querySelector("#generated");
const clearButton = document.querySelector("#clear");
const predictButton = document.querySelector("#predict");
const prediction = document.querySelector("#prediction");
const ctx = canvas.getContext("2d");
ctx.strokeStyle = "#000";
const CANVAS_SIZE = 28;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

mouseIsDown = false;
let lastX;
let lastY;

function draw(x, y, lastX, lastY) {
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.closePath();
  ctx.stroke();
}

function mouseDown(e) {
  mouseIsDown = true;

  const x = e.offsetX;
  const y = e.offsetY;

  lastX = x;
  lastY = y;
  mouseMove(e);
}

function mouseMove(e) {
  const x = e.offsetX;
  const y = e.offsetY;
  if (mouseIsDown) {
    draw(x, y, lastX, lastY);
  }

  lastX = x;
  lastY = y;
}

function mouseUp() {
  mouseIsDown = false;
}

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  prediction.innerHTML = "";
}

async function predictCanvas() {
  const model = await tf.loadLayersModel("classifier/model.json");
  const imgData = await ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  let filteredImgData = [];

  for (let i = 3; i < imgData.data.length; i = i + 4) {
    filteredImgData.push(imgData.data[i] / 255.0);
  }

  const tensor = tf.tensor([filteredImgData]);
  const reshapedTensor = tensor.reshape([1, 28, 28]);

  let predictions = await model.predict(reshapedTensor).array();

  predictions = predictions[0];
  let value = 0;
  let index;

  for (let i = 0; i < predictions.length; i++) {
    if (i === 0) {
      value = predictions[i];
      index = i;
    }

    if (predictions[i] > value) {
      value = predictions[i];
      index = i;
    }
  }

  prediction.innerHTML = index;
}

canvas.addEventListener("mousedown", mouseDown);
canvas.addEventListener("mousemove", mouseMove);
document.body.addEventListener("mouseup", mouseUp);
clearButton.addEventListener("click", clearCanvas);
predictButton.addEventListener("click", predictCanvas);
