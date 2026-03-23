import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';


const members = [
  { name: 'Saurabh Mazumdar', role: 'Lead Developer & Technical Strategist' },
  { name: 'Aryan Kumar', role: 'Full Stack & AI Systems Developer' },
  { name: 'Krishna Prakash', role: 'Machine Learning Engineer' },
  { name: 'Preesha Bhardwaj', role: 'UI/UX Designer & Frontend Developer' },
];


export default function Team() {
  return (
    <section className="glass-card p-6 sm:p-7">
      <div className="relative z-10">
        <p className="section-kicker">Meet the Team</p>
        <h2 className="section-title mt-2">Our Team</h2>
        <p className="section-copy mt-3 max-w-2xl">
          We are a multidisciplinary team passionate about applying artificial intelligence to improve
          early neurological diagnostics and healthcare accessibility.
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {members.map((member, index) => (
            <motion.article
              key={member.name}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.06 }}
              whileHover={{ y: -5, scale: 1.02 }}
              className="hover-lift rounded-3xl border border-ann-cyan/35 bg-gradient-to-b from-slate-900/82 to-slate-950/88 p-5 shadow-soft transition duration-300 hover:border-ann-cyan/60 hover:shadow-glow"
            >
              <div className="mb-4 inline-flex rounded-xl border border-slate-700/70 bg-slate-900/70 p-2">
                <Sparkles className="h-5 w-5 text-ann-cyan" />
              </div>
              <p className="font-display text-lg font-semibold text-white">{member.name}</p>
              <p className="mt-2 text-sm text-slate-300">{member.role}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}