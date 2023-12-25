
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_6_11>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	v_mov_b32_e32 v1, 0                                        // 00000000171C: 7E020280
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001720: 8488820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_ashr_i32 s3, s2, 31                                      // 000000001730: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[10:11], s[2:3], 2                             // 000000001738: 848A8202
	s_add_u32 s0, s0, s10                                      // 00000000173C: 80000A00
	s_addc_u32 s1, s1, s11                                     // 000000001740: 82010B01
	s_load_b32 s3, s[6:7], null                                // 000000001744: F40000C3 F8000000
	s_load_b32 s6, s[0:1], null                                // 00000000174C: F4000180 F8000000
	s_mul_i32 s0, s2, 11                                       // 000000001754: 96008B02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001758: BF870499
	s_ashr_i32 s1, s0, 31                                      // 00000000175C: 86019F00
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001760: 84808200
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001764: BF870009
	s_add_u32 s0, s4, s0                                       // 000000001768: 80000004
	s_addc_u32 s1, s5, s1                                      // 00000000176C: 82010105
	s_add_u32 s0, s0, s8                                       // 000000001770: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001774: 82010901
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_mul_f32_e64 v0, s3, s6                                   // 00000000177C: D5080000 00000C03
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001784: BF870001
	v_max_f32_e32 v0, 0, v0                                    // 000000001788: 20000080
	global_store_b32 v1, v0, s[0:1]                            // 00000000178C: DC6A0000 00000001
	s_nop 0                                                    // 000000001794: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001798: BFB60003
	s_endpgm                                                   // 00000000179C: BFB00000
