"use client"

import { useEffect, useRef, useState } from "react"
import { MainNav } from "@/components/main-nav"
import { siteConfig } from "@/config/site"
import Avatar, { genConfig } from "react-nice-avatar"
import { motion } from "framer-motion"
import QRCode from "react-qr-code"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Check, Copy, Loader2, Users, VideoOff, X } from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { Onboarding } from "@/components/onboarding"
import { useOnboarding } from "@/hooks/use-onboarding"
import { cn } from "@/lib/utils"
import { predictImage } from "@/api/requests"
import { baseUrl } from "@/services/client"
import RoundSelector from "@/components/round-selector"
import ReadyTimer from "@/components/ready-timer"

const copyLink = async (currentURL: string) => {
  await navigator.clipboard.writeText(currentURL)
  toast({
    title: "Link copied!",
    description: "The invite link has been copied to your clipboard.",
  })
}

export default function PlayPage() {
  const [currentRound, setCurrentRound] = useState(0)
  const [totalRounds, setTotalRounds] = useState<number | null>()
  const [player2, setPlayer2] = useState(null)
  const [id, setId] = useState<string | null>(null)
  useEffect(() => {
    if (window !== undefined) {
      setId(window.location.pathname.split('/')[1])
    }
  }, []);
  const currentPlayer = localStorage.getItem("playerName") ?? "You"
  const currentURL = window.location.href
  const [havePermission, setHavePermission] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [opponentFrame, setOpponentFrame] = useState(null)
  const [myFrame, setMyFrame] = useState(null)
  const [gameState, setGameState] = useState<
    'waiting' | 'toturial' | 'rounds_selection' | 'rounds_decision' | 'playing' | 
    'calculating_round_winner' | 'announcing_round_winner' | 'game_over'
  >('toturial');
  const [scores, setScores] = useState({you: 0, opponent: 0})

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream

          setVideoDimensions({
            width: videoRef.current.offsetWidth,
            height: videoRef.current.offsetHeight,
          });
        }
        setHavePermission(true)
      } catch (err) {
        setHavePermission(false)
      }
    }

    startCamera()

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream
        const tracks = stream.getTracks()
        tracks.forEach((track) => track.stop())
      }
    }
  }, [havePermission])

  const [socket, setSocket] = useState<WebSocket | null>(null)
  const [player2CompleteTask, setPlayer2CompleteTask] = useState<boolean>(false);
  const [probabilities, setProbabilities] = useState({ paper: 0, rock: 0, scissors: 0 });
  const [stage, setStage] = useState<"ready" | "waiting_ready" | "countdown" | "action" | "done">("ready")
  const [roundWinner, setRoundWinner] = useState<"you" | "opponent" | "draw">("draw")
  const [roundMoves, setRoundMoves] = useState<{ you: string, opponent: string }>({ you: "", opponent: "" })
  const [isHandMoving, setIsHandMoving] = useState<boolean | null>()
  const [cheating, setCheating] = useState<{is_cheated: boolean, cheater: string}>({ is_cheated: false, cheater: "" })
  useEffect(() => {
    if (!id) return
    if (socket) {
      socket.close()
    }
    const ws = new WebSocket(`ws://${baseUrl}:8000/ws/game/${id}/`)
    setSocket(ws)

    ws.onopen = () => {
      ws.send(JSON.stringify({ player_name: currentPlayer, type: "player_joined" }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.player_name !== currentPlayer) {
        setPlayer2(data.player_name)
      }
      if (data.type === "frame" && data.frame) {
        if (data.player_name !== currentPlayer) setOpponentFrame(data.frame)
        if (data.player_name === currentPlayer) {
          setMyFrame(data.frame)
          setIsHandMoving(data.is_moving)
        }
      } else if (data.type === "predict" && data.player_name === currentPlayer && data.predictions) {
        setProbabilities(data.predictions)
      } else if (data.type === "player_joined" && data.player_name !== currentPlayer) {
        setPlayer2(data.player_name)
      } else if (data.type === 'task' && data.player_name !== currentPlayer && data.task === 'scissor') {
        setPlayer2CompleteTask(true);
      } else if (data.type === 'rounds') {
        setTotalRounds(data.final_round)
      } else if (data.type === 'round_answer') {
        if (data.winner === 'draw') {
          setRoundWinner('draw')
        } else if (data.winner === currentPlayer) {
          setRoundWinner('you')
        } else {
          setRoundWinner('opponent')
        }
        let newScores = { ...scores }
        for (const [key, value] of Object.entries(data.scores)) {
          if (key === currentPlayer) {
            newScores.you = value as number
          } else {
            newScores.opponent = value as number
          }
        }
        setScores(newScores)
        console.log("Scores: ", newScores)
        for (let i = 0; i < data.moves.length; i++) {
          if (data.moves[i].player === currentPlayer) {
            setRoundMoves((prev) => ({ ...prev, you: data.moves[i].move }))
          } else {
            setRoundMoves((prev) => ({ ...prev, opponent: data.moves[i].move }))
          }
        }
        setOpponentReady(false)
        setGameState('announcing_round_winner')
      } else if (data.type === 'ready' && data.player_name !== currentPlayer) {
        setOpponentReady(true)
      } else if (data.type === 'cheating') {
        console.log("Cheating detected!")
        let newScores = { ...scores }
        for (const [key, value] of Object.entries(data.scores)) {
          if (key === currentPlayer) {
            newScores.you = value as number
          } else {
            newScores.opponent = value as number
          }
        }
        setScores(newScores)
        console.log("Scores: ", newScores)
        setCalculateMovement(false)
        setCheating({ is_cheated: true, cheater: data.cheater })
      }
    }
    ws.onclose = (event) => {
        console.warn(`WebSocket closed: Code=${event.code}, Reason=${event.reason}`);
    };  

    return () => {
      ws.close()
    }
  }, [id, currentPlayer])

  const [boxPosition, setBoxPosition] = useState({ top: 0, left: 0 });
  const [boxSize] = useState({ width: 100, height: 100 });
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [showCrown, setShowCrown] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [calculateMovement, setCalculateMovement] = useState(false);
  const FPS = 10;

  useEffect(() => {
    const captureAndSendFrame = async () => {
      if (videoRef.current && canvasRef.current && socket && socket.readyState === WebSocket.OPEN) {
        const video = videoRef.current
        const canvas = canvasRef.current
        const context = canvas.getContext("2d")

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        if (context)
          context.drawImage(video, 0, 0, canvas.width, canvas.height)

        const frameData = canvas.toDataURL("image/jpeg", 0.5)

        socket.send(JSON.stringify({ 
          frame: frameData, 
          show_crown: showCrown, 
          show_mask: showMask,
          calculate_movement: calculateMovement,
          player_name: currentPlayer, 
          type: "image_update" 
        }))
      }
    }

    const interval = setInterval(captureAndSendFrame, 1000 / FPS)

    return () => clearInterval(interval)
  }, [socket, showCrown, showMask, calculateMovement])

  const [shouldPredict, setShouldPredict] = useState(false)
  useEffect(() => {
    const captureAndSendFrame = async () => {
      if (videoRef.current && canvasRef.current && shouldPredict && id) {
        const video = videoRef.current
        const canvas = canvasRef.current
        const context = canvas.getContext("2d")

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        if (context)
          context.drawImage(video, 0, 0, canvas.width, canvas.height)

        const frameData = canvas.toDataURL("image/jpeg", 0.5)

        try {
          const predictions = await predictImage({ image: frameData, name: currentPlayer, round: currentRound.toString(), game: id });
          if (predictions) {
            setProbabilities(predictions)
          }
        } catch (error) {
          console.log(error);
        }
      }
    }

    const interval = setInterval(captureAndSendFrame, 1000 / FPS)

    return () => clearInterval(interval)
  }, [shouldPredict, currentPlayer, currentRound, id])

  const [roundPredictions, setRoundPredictions] = useState<{ paper: number, rock: number, scissors: number }[]>([]);
  const [timer, setTimer] = useState(3)
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    if (stage === "action") {
      setRoundPredictions((prev) => [...prev, probabilities])
      console.log("Predictions: ", probabilities)
    } else if (stage === "countdown" && isHandMoving === false && timer < 2) {
      console.log("You are not moving!")
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ player_name: currentPlayer, type: "cheating" }))
      }
    }
  }, [stage, probabilities, isHandMoving, timer, socket])

  useEffect(() => {
    if (stage === "done" && socket && socket.readyState === WebSocket.OPEN) {
      const paper = roundPredictions.reduce((acc, curr) => acc + curr.paper, 0) / roundPredictions.length
      const rock = roundPredictions.reduce((acc, curr) => acc + curr.rock, 0) / roundPredictions.length
      const scissors = roundPredictions.reduce((acc, curr) => acc + curr.scissors, 0) / roundPredictions.length
      const finalAnswer = paper > rock && paper > scissors ? "paper" : rock > paper && rock > scissors ? "rock" : "scissors"

      const firstHalf = roundPredictions.slice(0, Math.floor(roundPredictions.length / 2))
      const secondHalf = roundPredictions.slice(Math.floor(roundPredictions.length / 2));
      const firstHalfPaper = firstHalf.reduce((acc, curr) => acc + curr.paper, 0) / firstHalf.length
      const firstHalfRock = firstHalf.reduce((acc, curr) => acc + curr.rock, 0) / firstHalf.length
      const firstHalfScissors = firstHalf.reduce((acc, curr) => acc + curr.scissors, 0) / firstHalf.length
      const secondHalfPaper = secondHalf.reduce((acc, curr) => acc + curr.paper, 0) / secondHalf.length
      const secondHalfRock = secondHalf.reduce((acc, curr) => acc + curr.rock, 0) / secondHalf.length
      const secondHalfScissors = secondHalf.reduce((acc, curr) => acc + curr.scissors, 0) / secondHalf.length
      const firstHalfAnswer = firstHalfPaper > firstHalfRock && firstHalfPaper > firstHalfScissors ? "paper" : firstHalfRock > firstHalfPaper && firstHalfRock > firstHalfScissors ? "rock" : "scissors"
      const secondHalfAnswer = secondHalfPaper > secondHalfRock && secondHalfPaper > secondHalfScissors ? "paper" : secondHalfRock > secondHalfPaper && secondHalfRock > secondHalfScissors ? "rock" : "scissors"
      if (firstHalfAnswer !== secondHalfAnswer) {
        console.log("Answer: cheated", "Round: ", currentRound)
        console.log("First Half: ", firstHalfAnswer, "Second Half: ", secondHalfAnswer)
        socket.send(JSON.stringify({ answer: 'cheated', round: currentRound, player_name: currentPlayer, type: "round_answer" }))
      } else {
        console.log("Answer: ", finalAnswer, "Round: ", currentRound)
        socket.send(JSON.stringify({ answer: finalAnswer, round: currentRound, player_name: currentPlayer, type: "round_answer" }))
      }
      setStage("ready")
      setGameState('calculating_round_winner')
      setCalculateMovement(false)
      setRoundPredictions([])
    }
  }, [stage, roundPredictions, socket, currentRound, currentPlayer])

  const [isAdded, setIsAdded] = useState(false);
  const [opponentReady, setOpponentReady] = useState(false);
  useEffect(() => {
    if (gameState === 'announcing_round_winner' && !isAdded) {
      if (roundWinner === 'you') {
        console.log("You won the round!")
        // setScores((prev) => ({ ...prev, you: prev.you + 1 }))
      } else if (roundWinner === 'opponent') {
        console.log("Oponnent won the round!")
        // setScores((prev) => ({ ...prev, opponent: prev.opponent + 1 }))
      } else if (roundWinner === 'draw') {
        console.log("Draw the round!")
        // setScores((prev) => ({ ...prev, you: prev.you + 1, opponent: prev.opponent + 1 }))
      }
      if (totalRounds) {
        if (currentRound < totalRounds)
          setCurrentRound((prev) => prev + 1)
        if (currentRound + 1 === totalRounds) {
          console.log(scores)
          if (scores.you > scores.opponent) {
            toast({
              title: "Congratulations",
              description: "You won the game!",
            })
            setShowCrown(true)
          } else if (scores.opponent > scores.you) {
            toast({
              title: "Game Over",
              description: "You lost the game!",
              variant: "destructive",
            })
          } else {
            toast({
              title: "Draw",
              description: "You both had same scores!",
            })
          }
        }
      }
      setIsAdded(true)
    }
  }, [gameState, roundWinner, currentRound, totalRounds, isAdded, scores])

  const [showTutorial, setShowTutorial] = useState<boolean>(true);

  const initialTasks = [
    {
      id: "rock",
      title: "Show some rocks! ✊",
      status: "loading" as const,
    },
    {
      id: "paper",
      title: "Where are your papers? 🖐️",
      status: "pending" as const,
    },
    {
      id: "scissor",
      title: "It's time for the final cut! ✌️",
      status: "pending" as const,
    },
  ]
  const { tasks, currentTaskId, completeTask } = useOnboarding(initialTasks)
  const [completedTasks, setCompletedTasks] = useState<number>(0);
  const acceptanceThreshold = 0.7;

  useEffect(() => {
    if (currentTaskId && completedTasks < initialTasks.length) {
      if ((currentTaskId === 'rock' && probabilities.rock > acceptanceThreshold) || 
          (currentTaskId === 'paper' && probabilities.paper > acceptanceThreshold) || 
          (currentTaskId === 'scissor' && probabilities.scissors > acceptanceThreshold)) {
        completeTask(currentTaskId)
        if (socket && socket.readyState === WebSocket.OPEN) {
          console.log("Task completed: ", currentTaskId)
          socket.send(JSON.stringify({ task: currentTaskId, player_name: currentPlayer, type: "task_complete" }))
        }
        setCompletedTasks((prev) => prev + 1);
      }
    }
  }, [currentTaskId, completeTask, probabilities])

  useEffect(() => {
    const updateVideoDimensions = () => {
      if (videoRef.current) {
        setVideoDimensions({
          width: videoRef.current.offsetWidth,
          height: videoRef.current.offsetHeight,
        });
      }
    };

    updateVideoDimensions();

    window.addEventListener("resize", updateVideoDimensions);
    return () => window.removeEventListener("resize", updateVideoDimensions);
  }, []);
  
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (currentTaskId === "box") {
        if (videoRef && videoRef.current) {
          setVideoDimensions({
            width: videoRef.current.offsetWidth,
            height: videoRef.current.offsetHeight,
          });
        }
        if (event.key === 'Enter') {
          completeTask(currentTaskId)
          if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify({ task: "box", player_name: currentPlayer, type: "task_compelte" }))
          }
          return;
        }
        setBoxPosition((prev) => {
          const step = 10;
          switch (event.key) {
            case "ArrowUp":
              return { ...prev, top: Math.max(prev.top - step, 0) };
            case "ArrowDown":
              return { ...prev, top: Math.min(prev.top + step, Math.max(0, videoDimensions.height - boxSize.height)) };
            case "ArrowLeft":
              return { ...prev, left: Math.max(prev.left - step, 0) };
            case "ArrowRight":
              return { ...prev, left: Math.min(prev.left + step, Math.max(0, videoDimensions.width - boxSize.width)) };
            default:
              return prev;
          }
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentTaskId, videoDimensions, boxSize]);

  const isOnline = socket && socket.readyState === WebSocket.OPEN;

  useEffect(() => {
    if (completedTasks === initialTasks.length && player2CompleteTask && showTutorial) {
      setShowTutorial(false);
      setGameState('rounds_selection');
    }
  }, [completedTasks, player2CompleteTask, initialTasks]);

  const handleRoundsConfirm = (rounds: number) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ rounds, player_name: currentPlayer, type: "rounds_set" }))
      setGameState('rounds_decision');
      console.log("Game State: ", gameState)
      setCurrentRound(0);
    } else {
      toast({
        title: "Error",
        description: "Failed to set rounds.",
      })
    }
  }

  const handleReady = () => {
    setStage("waiting_ready")
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ player_name: currentPlayer, round: currentRound, type: "ready" }))
    }
  }

  useEffect(() => {
    if (opponentReady && stage === 'waiting_ready') {
      setStage("countdown")
      setCalculateMovement(true)
      setTimer(3)
      setProgress(0)
    }

    if (stage === 'action' || gameState === 'toturial') {
      setShouldPredict(true)
    } else {
      setShouldPredict(false)
    }
  }, [opponentReady, stage, gameState])

  useEffect(() => {
    if (cheating.is_cheated) {
      setStage("ready")
      if (cheating.cheater === currentPlayer) {
        setRoundWinner('opponent')
        setShowMask(true)
      } else {
        setRoundWinner('you')
      }
      
      setOpponentReady(false)
      setGameState('announcing_round_winner')
    }
  }, [cheating])

  return (
    <div className="container">
      <div className="pb-5 px-10 justify-between flex">
        <MainNav items={siteConfig.mainNav} />
        <div className="flex items-end gap-2 text-gray-200">
          <div className="flex items-center gap-2">
            <div className="relative">
              <div className={`w-3 h-3 rounded-full ${isOnline ? "bg-green-500" : "bg-red-500"}`}></div>
              <div className={cn(`absolute top-0 left-0 w-3 h-3 rounded-full ${isOnline ? "bg-green-500" : "bg-red-500"}`, isOnline ? 'animate-ping' : '')}></div>
            </div>
            <span className="font-semibold">{isOnline ? "Online" : "Offline"}</span>
          </div>
        </div>
      </div>
      <motion.section
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="flex flex-col bg-white rounded-[5px] shadow-md overflow-hidden px-3"
      >
        <div className="flex flex-col md:flex-row">
          <div className={`flex flex-col w-full md:w-1/2 p-4 items-center justify-center`}>
            <div className="w-full flex gap-2 flex-col">
              <div className="flex flex-row gap-2 items-center">
                <Avatar className="w-11 h-11" {...genConfig(currentPlayer)} />
                <h3 className="text-lg font-bold">{currentPlayer}</h3>
              </div>
              {
                havePermission ? (
                  <div>
                      <div className="w-full h-max flex items-center justify-center drop-shadow-lg">
                        <video ref={videoRef} autoPlay muted className="w-full h-max object-cover rounded-[5px]" />
                        <canvas ref={canvasRef} style={{ display: "none" }} />
                        {
                          myFrame && (showCrown || showMask) &&
                          <div
                            style={{
                              position: "absolute",
                              top: `0px`,
                              left: `0px`,
                            }}
                          >
                            <img src={myFrame} alt="My frame" className="w-full h-full object-cover rounded-[5px]" />
                          </div>
                        }
                      </div>
                    {
                      gameState === 'toturial' && showTutorial && player2 &&
                      <div className="flex flex-col justify-center bg-gray-50">
                        <Onboarding tasks={tasks} currentTaskId={currentTaskId} />
                        {
                          completedTasks !== initialTasks.length ?
                          <div className="flex flex-row gap-5 items-center justify-center p-5 bg-gray-200 rounded-b-[5px]">
                            <div
                              className={cn(
                                "transition-all",
                                probabilities.rock > 0.8 ? "scale-150" : 
                                (probabilities.rock > 0.5 ? "scale-125" : "scale-75"),
                              )}
                            >
                              ✊
                            </div>
                            <div
                              className={cn(
                                "transition-all",
                                probabilities.paper > 0.8 ? "scale-150" : 
                                (probabilities.paper > 0.5 ? "scale-125" : "scale-75"),
                              )}
                            >
                              🖐️
                            </div>
                            <div
                              className={cn(
                                "transition-all",
                                probabilities.scissors > 0.8 ? "scale-150" : 
                                (probabilities.scissors > 0.5 ? "scale-125" : "scale-75"),
                              )}
                            >
                              ✌️
                            </div>
                          </div>
                          :
                          <div className="flex flex-row gap-5 items-center justify-center p-5 bg-emerald-300 rounded-b-[5px] text-emerald-600 font-bold">
                            You learned all moves!
                          </div>
                        }
                      </div>
                    }
                    {
                      gameState === 'rounds_selection' &&
                      <RoundSelector onChange={(e) => setCurrentRound(e)} onConfirm={handleRoundsConfirm}/>
                    }
                    {
                      gameState === 'rounds_decision' &&
                      <ReadyTimer setStage={setStage} stage={stage} timer={timer} setTimer={setTimer} progress={progress} setProgress={setProgress} handleReady={handleReady}/>
                    }
                  </div>
                ) : (
                  <div className="w-full h-64 flex flex-col items-center justify-center bg-gray-200 rounded-[5px]">
                    <VideoOff className="h-16 w-16 text-gray-500" />
                    <p className="text-lg font-bold text-gray-500">Camera not available</p>
                  </div>
                )
              }
            </div>
          </div>
          {player2 ? (
            <div className={`flex flex-col w-full md:w-1/2 p-4 items-center`}>
              <div className="w-full flex gap-2 flex-col">
                <div className="flex flex-row gap-2 items-center">
                  <Avatar className="w-11 h-11" {...genConfig(player2)} />
                  <h3 className="text-lg font-bold">{player2}</h3>
                </div>
                {
                  opponentFrame ? (
                    <div className="w-full h-fit flex items-center justify-center drop-shadow-lg">
                      <img src={opponentFrame} alt="Opponent's frame" className="w-full h-full object-cover rounded-[5px]" />
                    </div>
                  ) : (
                    <div className="w-full h-64 flex flex-col items-center justify-center bg-gray-200 rounded-[5px]">
                      <p className="text-lg font-bold text-gray-500">Waiting for opponent...</p>
                    </div>
                  )
                }
                </div>
                {
                  gameState === 'toturial' && showTutorial && player2 &&
                  <div className="flex flex-row items-center gap-2 justify-center bg-gray-50 h-full w-full text-center rounded-b-[5px]">
                    {
                      player2CompleteTask ?
                      <Check className="h-6 w-6 text-emerald-500" />
                      :
                      <Loader2 className="h-6 w-6 animate-spin text-gray-600/50" />
                    }
                    <h3 className="text-l font-bold text-gray-700">Learning moves!</h3>
                  </div>
                }
                {
                  gameState === 'rounds_selection' &&
                  <div className="flex flex-row items-center gap-2 justify-center bg-gray-50 h-full w-full text-center rounded-b-[5px]">
                    <Loader2 className="h-6 w-6 animate-spin text-gray-600/50" />
                    <h3 className="text-l font-bold text-gray-700">Selecting Rounds!</h3>
                  </div>
                }
                {
                  gameState === 'rounds_decision' && totalRounds && (stage === 'ready' || stage === 'waiting_ready') && (
                    !opponentReady ?
                    <div className="flex flex-row items-center gap-2 justify-center bg-gray-50 h-full w-full text-center rounded-b-[5px]">
                      <X className="h-6 w-6 text-red-600" />
                      <h3 className="text-l font-bold text-gray-700">Not Ready</h3>
                    </div>
                    :
                    <div className="flex flex-row items-center gap-2 justify-center bg-gray-50 h-full w-full text-center rounded-b-[5px]">
                      <Check className="h-6 w-6 text-emerald-500" />
                      <h3 className="text-l font-bold text-gray-700">Ready</h3>
                    </div>
                  )
                }
              </div>
          ) : (
            <div className="flex flex-col md:w-1/2 p-4 gap-6 border-dashed border h-min m-5 rounded-[5px]">
              <div className="flex flex-row items-center gap-2">
                <Users />
                <h2 className="font-bold text-xl md:text-2xl">Invite your friend</h2>
              </div>
              <div className="w-full items-center flex justify-center">
                <div className="p-3 rounded-[5px] h-min w-min bg-[#E2E8F0]">
                  <QRCode value={currentURL} size={160} bgColor="#E2E8F0" />
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="invite-link">Or share this invite link</Label>
                <div className="flex space-x-2">
                  <Input id="invite-link" value={currentURL} readOnly className="bg-white rounded-[5px]" />
                  <Button variant="outline" size="icon" onClick={() => copyLink(currentURL)} className="rounded-[5px]">
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>
        {
          gameState === 'rounds_decision' && !totalRounds &&
          <div className="flex flex-row items-center gap-2 justify-center bg-gray-50 h-full w-full text-center p-5 rounded-t-[5px]">
            <Loader2 className="h-6 w-6 animate-spin text-gray-600/50" />
            <h3 className="text-l font-bold text-gray-700">Deciding Rounds!</h3>
          </div>
        }
        {
          gameState === 'calculating_round_winner' &&
          <div className="flex flex-row items-center gap-2 justify-center h-full w-full text-center p-5 rounded-t-[5px]">
            <Loader2 className="h-6 w-6 animate-spin text-gray-600/50" />
            <h3 className="text-l font-bold text-gray-700">Calculating Round Winner</h3>
          </div>
        }
        {
          gameState === 'announcing_round_winner' &&
          <div className="flex flex-col items-center gap-2 justify-center bg-gray-50 h-full w-full text-center p-5 rounded-t-[5px]">
            <h3 className="text-l font-bold text-gray-700">
              {roundWinner === 'you' ? "You won!" : roundWinner === 'opponent' ? "Opponent won!" : "It's a draw!"}
            </h3>
            {
              cheating.is_cheated ?
              <div className="flex flex-row items-center gap-2 justify-center">
                <X className="h-6 w-6 text-red-600" />
                <h3 className="text-l font-bold text-red-600">Cheating Detected!</h3>
              </div>
              :
              <div className="flex flex-row gap-2 items-center">
                <div className="transition-all">
                  {roundMoves.you === 'rock' ? "✊" : roundMoves.you === 'paper' ? "🖐️" : "✌️"}
                </div>
                <div className="text-sm font-bold">
                  Vs
                </div>
                <div className="transition-all">
                  {roundMoves.opponent === 'rock' ? "✊" : roundMoves.opponent === 'paper' ? "🖐️" : "✌️"}
                </div>
              </div>
            }
            {
              totalRounds && currentRound < totalRounds &&
              <Button onClick={() => {
                setGameState('rounds_decision');
                setIsAdded(false);
                setCheating({is_cheated: false, cheater: ""})
                setShowMask(false);
              }} className="rounded-[5px]">Next Round</Button>
            }
          </div>
        }
        {(gameState === 'rounds_decision' || gameState === 'playing' || gameState === 'announcing_round_winner' || gameState === 'calculating_round_winner') && totalRounds && (
          <div className="bg-gray-200 p-4 text-center rounded-t-[5px]">
            <h3 className="text-2xl font-bold">{scores.you} - {scores.opponent}</h3>
            <h3 className="text-sm opacity-25 font-bold">Round {Math.min(currentRound + 1, totalRounds)} / {totalRounds}</h3>
          </div>
        )}
      </motion.section>
    </div>
  )
}