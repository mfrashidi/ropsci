"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MinusIcon, PlusIcon } from "lucide-react"

interface RoundSelectorProps {
  onChange: (rounds: number) => void
  onConfirm: (rounds: number) => void
}

export default function RoundSelector({ onChange, onConfirm }: RoundSelectorProps) {
  const [rounds, setRounds] = useState<number>(1)

  const isValidRound = (num: number): boolean => {
    return num >= 0 && num <= 9 && num % 2 !== 0
  }

  const updateRounds = (newRounds: number) => {
    if (isValidRound(newRounds)) {
      setRounds(newRounds)
      onChange(newRounds)
    }
  }

  const increment = () => {
    updateRounds(rounds + 2)
  }

  const decrement = () => {
    updateRounds(rounds - 2)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number.parseInt(e.target.value, 10)
    if (!isNaN(value)) {
      updateRounds(value)
    }
  }

  const handleConfirm = () => {
    if (isValidRound(rounds)) {
      onConfirm(rounds)
    }
  }

  return (
    <div className="flex flex-col items-center space-y-5 bg-gray-50 pt-5 rounded-b-[5px]">
      <label htmlFor="roundInput" className="text-md font-semibold">
        Number of Rounds?
      </label>
      <div className="flex items-center space-x-2">
        <Button onClick={decrement} disabled={rounds <= 1} aria-label="Decrease rounds" className="w-10 h-10 p-0 rounded-[5px] bg-[#FF8152] hover:bg-[#c3785d]">
          <MinusIcon className="h-4 w-4" />
        </Button>
        <Input
          id="roundInput"
          value={rounds}
          onChange={handleInputChange}
          min={0}
          max={9}
          step={2}
          className="w-16 text-center bg-white rounded-[5px]"
          aria-label="Enter number of rounds"
        />
        <Button onClick={increment} disabled={rounds >= 9} aria-label="Increase rounds" className="w-10 h-10 p-0 rounded-[5px] bg-[#FF8152] hover:bg-[#c3785d]">
          <PlusIcon className="h-4 w-4" />
        </Button>
      </div>
      {!isValidRound(rounds) && <p className="text-red-500 text-sm">Please enter an odd number between 0 and 9.</p>}
      <Button onClick={handleConfirm} disabled={!isValidRound(rounds)} className="mt-2 w-full rounded-[5px] font-bold bg-[#FF8152] hover:bg-[#c3785d]">
        Confirm Rounds
      </Button>
    </div>
  )
}

