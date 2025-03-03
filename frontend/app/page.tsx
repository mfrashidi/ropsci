'use client'

import { siteConfig } from "@/config/site"
import { MainNav } from "@/components/main-nav"
import { FadeText } from "@/components/ui/fade-text"
import WordFadeIn from "@/components/ui/word-fade-in"
import Marquee from "@/components/ui/marquee"
import { motion } from "framer-motion"
import Avatar, { genConfig } from 'react-nice-avatar'

import Rock1 from "@/assets/rocks/1.jpg"
import Rock2 from "@/assets/rocks/2.jpg"
import Rock3 from "@/assets/rocks/3.jpg"
import Rock4 from "@/assets/rocks/4.jpg"
import Rock5 from "@/assets/rocks/5.jpg"
import Rock6 from "@/assets/rocks/6.jpg"
import Rock7 from "@/assets/rocks/7.jpg"

import Paper1 from "@/assets/papers/1.jpg"
import Paper2 from "@/assets/papers/2.jpg"
import Paper3 from "@/assets/papers/3.jpg"
import Paper4 from "@/assets/papers/4.jpg"
import Paper5 from "@/assets/papers/5.jpg"
import Paper6 from "@/assets/papers/6.jpg"
import Paper7 from "@/assets/papers/7.jpg"

import Scissors1 from "@/assets/scissors/1.jpg"
import Scissors2 from "@/assets/scissors/2.jpg"
import Scissors3 from "@/assets/scissors/3.jpg"
import Scissors4 from "@/assets/scissors/4.jpg"
import Scissors5 from "@/assets/scissors/5.jpg"
import Scissors6 from "@/assets/scissors/6.jpg"
import Scissors7 from "@/assets/scissors/7.jpg"
import { Input } from "@/components/ui/input"
import { use, useEffect, useState } from "react"
import { InteractiveHoverButton } from "@/components/ui/interactive-hover-button"
import { InputOTP, InputOTPGroup, InputOTPSeparator, InputOTPSlot } from "@/components/ui/input-otp"

export default function IndexPage() {
  const [name, setName] = useState("");

  const rocks = [
    Rock1, Rock2, Rock3, Rock4, Rock5, Rock6, Rock7
  ];
  const papers = [
    Paper1, Paper2, Paper3, Paper4, Paper5, Paper6, Paper7
  ];
  const scissors = [
    Scissors1, Scissors2, Scissors3, Scissors4, Scissors5, Scissors6, Scissors7
  ];

  const generateRandomHexPath = () => {
    const randomHex = Math.floor(Math.random() * 16777215).toString(16);
    return `/${randomHex}`;
  }

  const playNewGame = () => {
    localStorage.setItem("playerName", name);
    window.location.href = `./${generateRandomHexPath()}`;
  }

  useEffect(() => {
    const name = localStorage.getItem("playerName");
    if (name) {
      setName(name);
    }
  }, []);

  const [value, setValue] = useState("")
  useEffect(() => {
    if (value.length === 6) {
      localStorage.setItem("playerName", name);
      window.location.href = `./${value}`
    }
  }, [value])

  return (
    <div className="container">
      <div className="pb-5 px-10">
        <MainNav items={siteConfig.mainNav} />
      </div>
      <section className="container flex flex-col md:flex-row gap-10 pb-8 pt-6 md:py-10 bg-white rounded-[5px] h-full shadow-md">
        <div className="flex flex-col gap-5">
          <div className="text-l font-bold text-black/30 dark:text-white">
            It&apos;s all about
          </div>
          <div className="w-max">
            <FadeText
              className="text-2xl font-bold text-black dark:text-white flex items-center gap-2"
              direction="up"
              framerProps={{
                show: { transition: { delay: 0.2 } },
              }}
            >
              ✊ Rocks
            </FadeText>
          </div>
          <div className="w-max">
            <FadeText
              className="text-2xl font-bold text-black dark:text-white flex items-center gap-2"
              direction="up"
              framerProps={{
                show: { transition: { delay: 0.6 } },
              }}
            >
              🖐️ Papers
            </FadeText>
          </div>
          <WordFadeIn words="and..." className="text-l font-bold text-black/30 dark:text-white" delay={1} />
          <div className="w-max">
            <FadeText
              className="text-2xl font-bold text-black dark:text-white flex items-center gap-2"
              direction="up"
              framerProps={{
                show: { transition: { delay: 1.4 } },
              }}
            >
              ✌️ Scissors
            </FadeText>
          </div>
        </div>
        <div className="gap-5 overflow-hidden hidden md:flex"
          style={{
            maskImage: 'linear-gradient(to top, transparent, black 10%, black 90%, transparent)'
          }}
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.8, duration: 1 }}
            className="max-h-10"
          >
            <Marquee pauseOnHover vertical className="[--duration:20s]">
              {rocks.map((rock) => (
                <img key={rock.src} src={rock.src} className="w-32 h-28 rounded-[5px] shadow-sm border border-black/20" />
              ))}
            </Marquee>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.2, duration: 1 }}
            className="max-h-10"
          >
            <Marquee pauseOnHover reverse vertical className="[--duration:20s]">
              {papers.map((paper) => (
                <img key={paper.src} src={paper.src} className="w-32 h-28 rounded-[5px] shadow-sm border border-black/20" />
              ))}
            </Marquee>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2.6, duration: 1 }}
            className="max-h-10"
          >
            <Marquee pauseOnHover vertical className="[--duration:20s]">
              {scissors.map((scissor) => (
                <img key={scissor.src} src={scissor.src} className="w-32 h-28 rounded-[5px] shadow-sm border border-black/20" />
              ))}
            </Marquee>
          </motion.div>
        </div>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 3, duration: 1 }}
          className="h-full"
        >
            <div className="w-max flex justify-between flex-col gap-6 items-center h-full">
              <Avatar className="w-24 h-24" {...genConfig(name) } />
              <div className="flex flex-col gap-2">
                <Input type="text" placeholder="Name" className="bg-white rounded-xl" onChange={(e) => setName(e.target.value)} value={name}/>
                <InteractiveHoverButton className="w-full rounded-xl text-white" onClick={() => playNewGame()}>New Game</InteractiveHoverButton>
                <h5 className="opacity-50">...or join</h5>
                <InputOTP maxLength={6} value={value} onChange={(value) => setValue(value)}>
                  <InputOTPGroup>
                    <InputOTPSlot index={0} />
                    <InputOTPSlot index={1} />
                    <InputOTPSlot index={2} />
                  </InputOTPGroup>
                  <InputOTPSeparator />
                  <InputOTPGroup>
                    <InputOTPSlot index={3} />
                    <InputOTPSlot index={4} />
                    <InputOTPSlot index={5} />
                  </InputOTPGroup>
                </InputOTP>
              </div>
            </div>
        </motion.div>
      </section>
    </div>
  )
}
