import {
  FaceDetector,
  FilesetResolver,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0'

class JellyVoice {
  constructor(outputElement) {
    this.outputElement = outputElement
  }

  speak() {
    const index = Math.floor(Math.random() * this.phrases.length)
    this.outputElement.innerText = this.phrases[index]
    console.log(this.phrases[index])
  }

  quieten() {
    this.outputElement.innerText = ''
  }

  phrases = `You’re doing great
    You look fantastic
    You are everything you need
    You are beautiful
    You look good. Really. I’ve done the math and 100% of the times I’ve seen you I’ve felt good about it.
    I’d fight for you. One of these fists has your name on it. In a good way.
    I hope the sun shines on you, regardless of the weather.
    You look like a strong swimmer.
    Nice outfit.
    I’ll never be as smart as you.
    May you be kind to yourself in this moment.
    Be gentle with yourself.
    Asking for help can be strong, and brave.
    Newtown is the best suburb!
    Give yourself as much self-compassion as you can.
    You’re doing so well, I see how hard you’re trying.
    You are cared for.
    You belong.
    You are valued.
    All of you can be accepted.
    You deserve to feel safe.
    You deserve to feel joy!
    Take a deep breath.
    Your body is a wonderful thing.
    It’s ok to feel your emotions.
    I love Newtown.
    This too shall pass.
    You can do it!
    I’m proud of you.
    You’re learning every day.
    One step at a time and you’ll get there!`
    .split('\n')
    .map((p) => p.trim())
}

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
    publishPosition(video, detections?.[0])
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
      detection.boundingBox.originY +
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

const jellyVoice = new JellyVoice(document.getElementById('voice'))
let previousFocusX = 0.5
let previousFocusY = 0.5
let focusedAmount = 0
const jellyEyes = document.getElementById('eyes')
function publishPosition(video, detection) {
  const x =
    0.5 -
    (detection?.boundingBox?.originX + detection?.boundingBox?.width / 2) /
      video.offsetWidth
  const y =
    (detection?.boundingBox?.originY + detection?.boundingBox?.height / 2) /
      video.offsetHeight -
    0.5

  if (x && y) {
    jellyEyes.style.left = `${50 * x}%`
    jellyEyes.style.top = `${20 + 15 * y}%`

    if (
      Math.abs(x - previousFocusX) < 0.1 &&
      Math.abs(y - previousFocusY) < 0.1
    ) {
      focusedAmount++
    }
    previousFocusX = x
    previousFocusY = y
    if (focusedAmount == 80) {
      jellyVoice.speak()
    }
  } else {
    jellyEyes.style.removeProperty('left')
    jellyEyes.style.removeProperty('top')
    focusedAmount = 0
    previousFocusX = NaN
    previousFocusY = NaN
    jellyVoice.quieten()
  }
}
