"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Check } from "lucide-react"

interface ReadyTimerProps {
    stage: "ready" | "waiting_ready" | "countdown" | "action" | "done"
    setStage: React.Dispatch<React.SetStateAction<"ready" | "waiting_ready" | "countdown" | "action" | "done">>
    timer: number
    setTimer: React.Dispatch<React.SetStateAction<number>>
    progress: number
    setProgress: React.Dispatch<React.SetStateAction<number>>
    handleReady: () => void
}

export default function ReadyTimer({ stage, setStage, timer, setTimer, progress, setProgress, handleReady }: ReadyTimerProps) {
  useEffect(() => {
    let interval: NodeJS.Timeout

    if (stage === "countdown" && timer > 0) {
      interval = setInterval(() => {
        setTimer((prevTimer) => prevTimer - 1)
      }, 1000)
    } else if (stage === "countdown" && timer === 0) {
      setStage("action")
      setTimer(1)
      setProgress(0)
    } else if (stage === "action") {
      interval = setInterval(() => {
        setProgress((prevProgress) => {
          if (prevProgress >= 100) {
            clearInterval(interval)
            setStage("done")
            return 100
          }
          return prevProgress + 1
        })
      }, 10)
    }

    return () => clearInterval(interval)
  }, [stage, timer])

//   const handleReady = () => {
//     setStage("countdown")
//     setTimer(3)
//     setProgress(0)
//   }

  return (
    <div className="flex flex-col items-center justify-center space-y-5 bg-gray-50 pt-5">
      <div className="rounded-lg text-center">
        {stage === "ready" && (
            <div>
                <h2 className="text-lg font-bold mb-4 px-8">Keep your hand up and hit ready whenever you are!</h2>
                <Button onClick={handleReady} className="w-full rounded-[5px] font-bold bg-[#FF8152] hover:bg-[#c3785d]">
                    I&apos;m ready
                </Button>
            </div>
        )}
        {stage === "waiting_ready" && (
            <div className="flex flex-row items-center gap-2 justify-centerh-full w-full text-center py-5">
                <Check className="h-6 w-6 text-emerald-500" />
                <h3 className="text-l font-bold text-gray-700">Ready</h3>
            </div>
        )}
        {stage === "countdown" && (
          <div className="mb-8">
            <h2 className="text-lg font-bold mb-4">Wave your hand 👋</h2>
            <p className="text-2xl font-bold">{timer}</p>
          </div>
        )}
        {stage === "action" && (
          <div className="mb-8">
            <h2 className="text-lg font-bold mb-4">Keep your hand steady!</h2>
            <div className="relative w-32 h-32 mx-auto">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                <circle
                  className="text-gray-200 stroke-current"
                  strokeWidth="10"
                  cx="50"
                  cy="50"
                  r="40"
                  fill="transparent"
                />
                <circle
                  className="text-[#FF8152] stroke-current"
                  strokeWidth="10"
                  strokeLinecap="round"
                  cx="50"
                  cy="50"
                  r="40"
                  fill="transparent"
                  strokeDasharray="251.2"
                  strokeDashoffset={251.2 * (1 - progress / 100)}
                  transform="rotate(-90 50 50)"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold">{Math.ceil((100 - progress) / 100)}</span>
              </div>
            </div>
          </div>
        )}
        {stage === "done" && (
          <div>
            <h2 className="text-lg font-bold mb-4">Time&apos;s up!</h2>
            {/* <Button onClick={handleReady} className="text-lg mt-4">
              Try again
            </Button> */}
          </div>
        )}
      </div>
    </div>
  )
}

