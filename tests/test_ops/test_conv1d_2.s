
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_7_5>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_mul_i32 s8, s2, 5                                        // 00000000171C: 96088502
	s_lshl_b64 s[16:17], s[14:15], 2                           // 000000001720: 8490820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s16                                      // 000000001728: 80061006
	s_addc_u32 s7, s7, s17                                     // 00000000172C: 82071107
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001738: 84888208
	s_add_u32 s0, s0, s8                                       // 00000000173C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001740: 82010901
	s_load_b128 s[8:11], s[0:1], null                          // 000000001744: F4080200 F8000000
	s_clause 0x1                                               // 00000000174C: BF850001
	s_load_b128 s[12:15], s[6:7], null                         // 000000001750: F4080303 F8000000
	s_load_b32 s3, s[6:7], 0x10                                // 000000001758: F40000C3 F8000010
	s_load_b32 s1, s[0:1], 0x10                                // 000000001760: F4000040 F8000010
	s_mul_i32 s0, s2, 7                                        // 000000001768: 96008702
	s_waitcnt lgkmcnt(0)                                       // 00000000176C: BF89FC07
	v_fma_f32 v0, s12, s8, 0                                   // 000000001770: D6130000 0200100C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001778: BF870091
	v_fmac_f32_e64 v0, s13, s9                                 // 00000000177C: D52B0000 0000120D
	v_fmac_f32_e64 v0, s14, s10                                // 000000001784: D52B0000 0000140E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000178C: BF870091
	v_fmac_f32_e64 v0, s15, s11                                // 000000001790: D52B0000 0000160F
	v_fmac_f32_e64 v0, s3, s1                                  // 000000001798: D52B0000 00000203
	s_ashr_i32 s1, s0, 31                                      // 0000000017A0: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000017A4: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017A8: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017AC: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000017B0: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000017B4: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017B8: 82010105
	s_add_u32 s0, s0, s16                                      // 0000000017BC: 80001000
	s_addc_u32 s1, s1, s17                                     // 0000000017C0: 82011101
	global_store_b32 v1, v0, s[0:1]                            // 0000000017C4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017CC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017D0: BFB60003
	s_endpgm                                                   // 0000000017D4: BFB00000
