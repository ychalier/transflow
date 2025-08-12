use_bpm 60

# --- Atmosphere Base ---
live_loop :drone do
  with_fx :reverb, mix: 0.6, room: 1 do
    with_fx :lpf, cutoff: 80 do
      synth :hollow, note: :e1, sustain: 8, release: 4, amp: 0.4
      synth :dark_ambience, note: :g1, sustain: 8, release: 4, amp: 0.3
    end
  end
  sleep 8
end

# --- Roaming Background Noise ---
live_loop :background_noise do
  with_fx :pan, pan: rrand(-0.8, 0.8) do
    with_fx :hpf, cutoff: rrand(70, 110) do
      with_fx :echo, phase: rrand(0.3, 0.8), decay: 4 do
        synth :noise, release: 2, amp: 0.1
      end
    end
  end
  sleep rrand(1.5, 4)
end

# --- Electrical Buzzes ---
live_loop :buzzes do
  sleep rrand(4, 12)
  with_fx :pan, pan: rrand(-0.6, 0.6) do
    with_fx :distortion, distort: 0.5 do
      with_fx :lpf, cutoff: rrand(70, 100) do
        synth :bnoise, release: rrand(0.3, 1.0), amp: 0.2
      end
    end
  end
end

# --- Portal Effect: Bass Wave Ondulating ---
live_loop :portal do
  sleep rrand(60, 540)
  with_fx :reverb, room: 1, mix: 0.7 do
    with_fx :slicer, phase: 0.125, mix: 0.4, pulse_width: 0.6 do
      synth :mod_fm,
        note: :e1,
        sustain: 4,
        release: 2,
        mod_range: 8,
        mod_phase: 0.25,
        depth: 3,
        amp: 0.6
    end
  end
end

# --- Metallic Pings (rare, spatial) ---
live_loop :metallic_pings do
  sleep rrand(12, 25)
  with_fx :pan, pan: rrand(-0.9, 0.9) do
    with_fx :echo, phase: rrand(0.2, 0.5), decay: 6 do
      synth :pretty_bell, note: rrand_i(48, 60), amp: 0.15, release: 3
    end
  end
end

# --- Deep Rumbles (distant machinery) ---
live_loop :deep_rumble do
  sleep rrand(20, 40)
  with_fx :lpf, cutoff: rrand(60, 80) do
    with_fx :distortion, distort: 0.2, mix: 0.3 do
      synth :dark_ambience, note: :e1, release: 6, amp: 0.3
    end
  end
end

# --- Ghostly Whooshes (passing energy waves) ---
live_loop :ghostly_whoosh do
  sleep rrand(15, 30)
  with_fx :pan, pan: rrand(-0.7, 0.7) do
    with_fx :reverb, room: 1, mix: 0.6 do
      synth :hollow, note: rrand_i(50, 65), attack: 0.5, sustain: 2, release: 3, amp: 0.2
    end
  end
end

# --- Alien Chirps ---
live_loop :alien_chirps do
  sleep rrand(18, 40)
  with_fx :pan, pan: rrand(-0.8, 0.8) do
    with_fx :echo, phase: rrand(0.15, 0.4), decay: 3 do
      synth :pluck, note: rrand_i(72, 84), release: 0.8, amp: 0.15
    end
  end
end

# --- Energy Sparks ---
live_loop :energy_sparks do
  sleep rrand(10, 25)
  with_fx :pan, pan: rrand(-0.6, 0.6) do
    with_fx :reverb, room: 0.8, mix: 0.5 do
      synth :beep, note: rrand_i(60, 76), release: 0.3, amp: 0.2
      sleep 0.1
      synth :beep, note: rrand_i(60, 76), release: 0.3, amp: 0.15
    end
  end
end

# --- Gravity Pulses ---
live_loop :gravity_pulses do
  sleep rrand(25, 50)
  with_fx :lpf, cutoff: rrand(50, 70) do
    with_fx :distortion, distort: 0.3, mix: 0.4 do
      synth :fm, note: :c1, release: 3, amp: 0.4
    end
  end
end

# --- Distant Metallic Echoes ---
live_loop :metallic_echoes do
  sleep rrand(20, 45)
  with_fx :pan, pan: rrand(-0.9, 0.9) do
    with_fx :echo, phase: 0.4, decay: 8 do
      synth :pretty_bell, note: rrand_i(55, 65), release: 4, amp: 0.18
    end
  end
end

# --- Dark Vibrations ---
live_loop :dark_vibrations do
  sleep rrand(12, 20)
  with_fx :lpf, cutoff: rrand(50, 65) do
    with_fx :slicer, phase: rrand(0.25, 0.5), mix: 0.3, pulse_width: 0.7 do
      synth :mod_fm,
        note: [:c1, :d1, :e1].choose,
        release: 6,
        depth: rrand(2, 5),
        mod_range: rrand(4, 8),
        mod_phase: rrand(0.15, 0.3),
        amp: 0.4
    end
  end
end

# --- Portal Entry Effect ---
define :portal_entry do
  with_fx :reverb, mix: 0.8, room: 1 do
    with_fx :echo, phase: 0.25, decay: 6 do
      # Shimmering chord burst
      synth :pretty_bell, note: chord(:e4, :minor7), release: 4, amp: 0.5
      synth :blade, note: :e5, attack: 0.1, sustain: 1, release: 3, amp: 0.4
      # Bright high sparkle
      sleep 0.2
      synth :pluck, note: :e6, release: 1.5, amp: 0.3
    end
  end
end

# --- Portal Exit Effect ---
define :portal_exit do
  with_fx :reverb, mix: 0.8, room: 1 do
    with_fx :echo, phase: 0.3, decay: 5 do
      # Descending shimmer
      synth :pretty_bell, note: chord(:e5, :minor7).reverse, release: 3, amp: 0.45
      synth :blade, note: :e4, attack: 0.05, sustain: 1, release: 2.5, amp: 0.35
      # Soft fade-out sparkle
      sleep 0.2
      synth :pluck, note: :e3, release: 1.5, amp: 0.25
    end
  end
end

# portal_entry
# portal_exit
