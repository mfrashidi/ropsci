import { Check, Loader2 } from "lucide-react"
import type { OnboardingProps } from "@/types/onboarding"

export function Onboarding({ tasks, currentTaskId, onComplete }: OnboardingProps) {
  return (
    <div className="w-full max-w-md space-y-8 p-6">
      {tasks.map((task) => (
        <div
          key={task.id}
          className={`flex items-center space-x-4 transition-opacity ${
            task.status === "pending" ? "opacity-50" : "opacity-100"
          }`}
        >
          <div className="flex h-6 w-6 items-center justify-center">
            {task.status === "completed" ? (
              <Check className="h-6 w-6 text-emerald-500" />
            ) : task.status === "loading" ? (
              <Loader2 className="h-6 w-6 animate-spin text-gray-600/50" />
            ) : (
              <div className="h-6 w-6 rounded-full border-2 border-gray-300" />
            )}
          </div>
          <div className="space-y-1">
            <h3 className="text-l font-bold text-gray-700">{task.title}</h3>
            {
                task.subtitle &&
                <p className="text-gray-500 text-sm">{task.subtitle}</p>
            }
          </div>
        </div>
      ))}
    </div>
  )
}

