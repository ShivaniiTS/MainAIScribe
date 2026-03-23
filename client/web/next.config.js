/**
 * next.config.js — enable standalone output so Dockerfile can copy .next/standalone
 *
 * This config tells Next.js to produce a self-contained `./.next/standalone`
 * directory (server bundle + minimal node_modules) which the Dockerfile expects.
 */

/** @type {import('next').NextConfig} */
module.exports = {
  output: 'standalone',
};
