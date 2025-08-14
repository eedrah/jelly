import {
  FaceDetector,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0'

if (!navigator.mediaDevices?.getUserMedia) {
  alert('getUserMedia() is not supported by your browser')
}

let video = document.getElementById('webcam')
const liveView = document.getElementById('liveView')

let faceDetector
const initializefaceDetector = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
  )
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
  })

  navigator.mediaDevices
    .getUserMedia({
      video: true,
    })
    .then(function (stream) {
      video.srcObject = stream
      video.addEventListener('loadeddata', predictWebcam)
    })
    .catch((err) => {
      console.error(err)
    })
}
initializefaceDetector()

let lastVideoTime = -1
async function predictWebcam() {
  let startTimeMs = performance.now()

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime
    const detections = faceDetector.detectForVideo(
      video,
      startTimeMs
    ).detections
    displayVideoDetections(detections)
  }

  // repeat when ready
  window.requestAnimationFrame(predictWebcam)
}

var childrenToRemove = []
function displayVideoDetections(detections /* : Detection[] */) {
  for (let child of childrenToRemove) {
    liveView.removeChild(child)
  }
  childrenToRemove.splice(0)

  for (let detection of detections) {
    console.log(detection)
    const p = document.createElement('div')
    p.classList.add('data')
    p.innerText =
      'Confidence: ' +
      Math.round(parseFloat(detection.categories[0].score) * 100) +
      '% .'
    p.style =
      'left: ' +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      'px;' +
      'top: ' +
      (detection.boundingBox.originY - 30) +
      'px; ' +
      'width: ' +
      (detection.boundingBox.width - 10) +
      'px;'

    const highlighter = document.createElement('div')
    highlighter.setAttribute('class', 'highlighter')
    highlighter.style =
      'left: ' +
      (video.offsetWidth -
        detection.boundingBox.width -
        detection.boundingBox.originX) +
      'px;' +
      'top: ' +
      detection.boundingBox.originY +
      'px;' +
      'width: ' +
      (detection.boundingBox.width - 10) +
      'px;' +
      'height: ' +
      detection.boundingBox.height +
      'px;'

    liveView.appendChild(highlighter)
    liveView.appendChild(p)

    childrenToRemove.push(highlighter)
    childrenToRemove.push(p)
    for (let keypoint of detection.keypoints) {
      const keypointEl = document.createElement('span')
      keypointEl.className = 'key-point'
      keypointEl.style.top = `${keypoint.y * video.offsetHeight - 3}px`
      keypointEl.style.left = `${
        video.offsetWidth - keypoint.x * video.offsetWidth - 3
      }px`
      liveView.appendChild(keypointEl)
      childrenToRemove.push(keypointEl)
    }
  }
}
