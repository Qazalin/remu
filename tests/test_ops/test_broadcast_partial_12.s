
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_32_1024n3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s3, s15, 10                                     // 000000001714: 84038A0F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_add_i32 s8, s3, s14                                      // 00000000171C: 81080E03
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001720: BF870499
	s_ashr_i32 s9, s8, 31                                      // 000000001724: 86099F08
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001728: 84888208
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001730: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001734: 82070907
	s_ashr_i32 s3, s15, 31                                     // 000000001738: 86039F0F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001740: 84828202
	s_add_u32 s0, s0, s2                                       // 000000001744: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001748: 82010301
	s_load_b32 s2, s[6:7], null                                // 00000000174C: F4000083 F8000000
	s_load_b32 s0, s[0:1], null                                // 000000001754: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 00000000175C: BF89FC07
	v_div_scale_f32 v0, null, s0, s0, s2                       // 000000001760: D6FC7C00 00080000
	v_div_scale_f32 v3, vcc_lo, s2, s0, s2                     // 000000001768: D6FC6A03 00080002
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)// 000000001770: BF8700B2
	v_rcp_f32_e32 v1, v0                                       // 000000001774: 7E025500
	s_waitcnt_depctr 0xfff                                     // 000000001778: BF880FFF
	v_fma_f32 v2, -v0, v1, 1.0                                 // 00000000177C: D6130002 23CA0300
	v_fmac_f32_e32 v1, v2, v1                                  // 000000001784: 56020302
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001788: BF870091
	v_mul_f32_e32 v2, v3, v1                                   // 00000000178C: 10040303
	v_fma_f32 v4, -v0, v2, v3                                  // 000000001790: D6130004 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001798: BF870091
	v_fmac_f32_e32 v2, v4, v1                                  // 00000000179C: 56040304
	v_fma_f32 v0, -v0, v2, v3                                  // 0000000017A0: D6130000 240E0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)// 0000000017A8: BF870121
	v_div_fmas_f32 v0, v0, v1, v2                              // 0000000017AC: D6370000 040A0300
	v_mov_b32_e32 v1, 0                                        // 0000000017B4: 7E020280
	v_div_fixup_f32 v0, v0, s0, s2                             // 0000000017B8: D6270000 00080100
	s_add_u32 s0, s4, s8                                       // 0000000017C0: 80000804
	s_addc_u32 s1, s5, s9                                      // 0000000017C4: 82010905
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C8: DC6A0000 00000001
	s_nop 0                                                    // 0000000017D0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D4: BFB60003
	s_endpgm                                                   // 0000000017D8: BFB00000
